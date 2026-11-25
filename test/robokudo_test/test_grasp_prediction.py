"""
Grasps predicted on the point clouds of objects and handed to their graspable
annotations.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
import open3d
import py_trees
import pytest
import torch
from typing_extensions import Self

pytest.importorskip("graspgenx")

import robokudo.behaviours.clear_annotator_outputs
import robokudo.pipeline
import robokudo.utils.tree_execution
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence, StatementNode
from krrood.entity_query_language.factories import a, variable
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.grasp_pose_annotator import GraspPoseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    WorldDescriptorSource,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.grasp_prediction import GraspGenXGripper
from robokudo.types.annotation import GraspAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_transform_matrix_from_q
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Camera, RobotCamera
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ..coraplex_test.test_designator.test_multi_stationary_robot_action_designator import (
    graspable_annotation,
    robot_setup,
    stationary_block_context,
)
from ..plan_running import run_plan

# %% fixtures


@pytest.fixture(autouse=True)
def seeded_sampling():
    """
    Seed the noise the grasp network samples its grasps from, so a test sees the same
    grasps on every run.
    """
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def tracy_gripper(tracy_world) -> GraspGenXGripper:
    """
    Tracy's left gripper, a Robotiq 2F-85, as the grasp network is conditioned on it.
    """
    tracy = tracy_world.get_semantic_annotations_by_type(Tracy)[0]
    return GraspGenXGripper.from_two_finger_gripper(gripper=tracy.left_arm.end_effector)


@pytest.fixture
def can_on_the_table(stationary_block_context):
    """
    Tracy's world with a soda can standing on its table.
    """
    world, robot, extensions = stationary_block_context
    if not isinstance(robot, Tracy):
        pytest.skip("the gripper the grasps are predicted for is Tracy's")
    can_shape = Cylinder(width=0.066, height=0.12)
    can = Body(
        name=PrefixedName("can"),
        collision=ShapeCollection([can_shape]),
        visual=ShapeCollection([can_shape]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=can,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.8, 0.3, robot.table.top_z + can_shape.height / 2
                ),
            )
        )
        world.add_semantic_annotations([IsGraspable(root=can)])
    return world, robot, extensions, can


def optical_frame_of(camera: RobotCamera) -> HomogeneousTransformationMatrix:
    """
    :param camera: Tracy's camera.
    :return: The optical frame its driver publishes, in the world: its link looks along
        its x-axis, the optical frame along its z-axis.
    """
    return HomogeneousTransformationMatrix(
        camera.root.global_transform.to_np()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=-np.pi / 2, yaw=-np.pi / 2
        ).to_np(),
        reference_frame=camera.root._world.root,
    )


@dataclass(kw_only=True)
class CameraView:
    """
    What a camera sees: the nearest point on each of its rays and the body it lies on.
    """

    points: np.ndarray
    """
    The points, one per row, in the camera's optical frame.
    """

    bodies: list[Body]
    """
    The body each point lies on.
    """

    @classmethod
    def from_camera(cls, camera: RobotCamera) -> Self:
        """
        :param camera: Tracy's camera.
        :return: What it sees past its own housing.
        """
        world = camera.root._world
        world_T_optical = optical_frame_of(camera).to_np()
        ray_tracer = RayTracer(world)
        rays = ray_tracer.create_camera_rays(
            HomogeneousTransformationMatrix(
                world_T_optical @ Camera.optical_R_forward_view.to_np(),
                reference_frame=world.root,
            ),
            camera_model=PinholeCameraModel.from_field_of_view(
                resolution=camera.camera_model.resolution,
                field_of_view=camera.camera_model.field_of_view,
            ),
        )
        hits = ray_tracer.ray_test(
            rays.origin_points,
            rays.origin_points + rays.direction_vectors,
            multiple_hits=True,
        )
        outside_housing = np.array([hit is not camera.root for hit in hits.bodies])
        ray_indices = hits.ray_indices[outside_housing]
        points = hits.hit_points[outside_housing]
        bodies = [hit for hit in hits.bodies if hit is not camera.root]
        distances = np.linalg.norm(points - rays.origin_points[ray_indices], axis=1)
        by_ray_then_distance = np.lexsort((distances, ray_indices))
        nearest = by_ray_then_distance[
            np.unique(ray_indices[by_ray_then_distance], return_index=True)[1]
        ]
        optical_T_world = np.linalg.inv(world_T_optical)
        return cls(
            points=points[nearest] @ optical_T_world[:3, :3].T + optical_T_world[:3, 3],
            bodies=[bodies[index] for index in nearest],
        )

    def cloud(self, body: Body | None = None) -> open3d.geometry.PointCloud:
        """
        :param body: The body whose points to take, every body's if ``None``.
        :return: The points as a point cloud.
        """
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(
            self.points
            if body is None
            else self.points[[seen is body for seen in self.bodies]]
        )
        return cloud


def annotate_what_the_camera_sees_of(
    body: Body, camera: RobotCamera, gripper: GraspGenXGripper
) -> ObjectHypothesis:
    """
    Run the grasp pose annotator on what ``camera`` sees, with a hypothesis of ``body``
    matched to ``body`` as a perception pipeline matches it.

    :return: The annotated hypothesis.
    """
    view = CameraView.from_camera(camera)
    hypothesis = ObjectHypothesis(points=view.cloud(body), object_knowledge=body)
    descriptor = GraspPoseAnnotator.Descriptor()
    descriptor.parameters.gripper = gripper
    annotator = GraspPoseAnnotator(descriptor=descriptor)
    pipeline = robokudo.pipeline.Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.cloud = view.cloud()
    pipeline.cas.camera_observation = CameraObservation(
        camera=camera,
        effective_camera_model=PinholeCameraModel.from_field_of_view(
            resolution=camera.camera_model.resolution,
            field_of_view=camera.camera_model.field_of_view,
        ),
        world_T_camera=optical_frame_of(camera),
        timestamp_nanoseconds=0,
    )
    pipeline.cas.annotations.append(hypothesis)
    annotator.update()
    return hypothesis


# %% annotating the objects a pipeline perceives


def test_every_perceived_object_is_annotated_with_grasps_holding_it(
    node, tracy_gripper
):
    """
    Each object on the ray-traced tabletop is annotated with the most confident grasps
    predicted on its cluster, and the most confident one puts the cluster between the
    open fingertips.
    """
    raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
        RayTracingCameraConfiguration.registry_name,
        source=WorldDescriptorSource(descriptor_name="world_semdt_raytracer_tabletop"),
    )
    plane_descriptor = PlaneAnnotator.Descriptor()
    plane_descriptor.parameters.distance_threshold = 0.01
    plane_descriptor.parameters.random_seed = 0
    cluster_descriptor = PointCloudClusterExtractor.Descriptor()
    cluster_descriptor.parameters.dbscan_min_cluster_count = 8
    cluster_descriptor.parameters.min_cluster_count = 20
    cluster_descriptor.parameters.min_on_plane_point_count = 10
    cluster_descriptor.parameters.eps = 0.05
    grasp_descriptor = GraspPoseAnnotator.Descriptor()
    grasp_descriptor.parameters.gripper = tracy_gripper
    pipeline = robokudo.pipeline.Pipeline("GraspPosePipeline")
    pipeline.add_children(
        [
            robokudo.behaviours.clear_annotator_outputs.ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=raytracer_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(),
            PlaneAnnotator(descriptor=plane_descriptor),
            PointCloudClusterExtractor(descriptor=cluster_descriptor),
            GraspPoseAnnotator(descriptor=grasp_descriptor),
        ]
    )

    result = robokudo.utils.tree_execution.run_tree_once(
        pipeline, node, max_iterations=400, tick_rate=20
    )

    assert result is py_trees.common.Status.SUCCESS
    hypotheses = pipeline.cas.filter_annotations_by_type(ObjectHypothesis)
    assert len(hypotheses) == 4
    open_space = tracy_gripper.open_finger_space
    for hypothesis in hypotheses:
        grasps = pipeline.cas.filter_by_type(GraspAnnotation, hypothesis.annotations)
        confidences = [grasp.confidence for grasp in grasps]
        assert len(grasps) == grasp_descriptor.parameters.maximum_grasps
        assert confidences == sorted(confidences, reverse=True)
        camera_T_hand = get_transform_matrix_from_q(
            grasps[0].rotation, grasps[0].translation
        ) @ np.linalg.inv(tracy_gripper.hand_T_grasp)
        hand_T_camera = np.linalg.inv(camera_T_hand)
        points = np.asarray(hypothesis.points.points)
        hand_points = points @ hand_T_camera[:3, :3].T + hand_T_camera[:3, 3]
        between_fingertips = np.all(
            np.abs(hand_points - open_space.center) <= open_space.extents / 2, axis=1
        )
        assert between_fingertips.any()


# %% handing the grasps to the graspable


def test_the_grasps_of_a_matched_object_are_handed_to_its_graspable(
    stationary_block_context, tracy_gripper, annotator_output_buffer
):
    """
    The grasps annotated in the camera frame become the graspable's perceived grasps,
    standing at the same place in the world.
    """
    world, robot, _ = stationary_block_context
    if not isinstance(robot, Tracy):
        pytest.skip("the gripper the grasps are predicted for is Tracy's")
    box = world.get_body_by_name("box1")
    camera = robot.get_default_camera()

    hypothesis = annotate_what_the_camera_sees_of(box, camera, tracy_gripper)

    annotations = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, GraspAnnotation)
    ]
    perceived = graspable_annotation(world, box).perceived_grasps
    assert len(annotations) == GraspPoseAnnotator.Descriptor().parameters.maximum_grasps
    assert [grasp.confidence for grasp in perceived] == [
        annotation.confidence for annotation in annotations
    ]
    world_T_camera = optical_frame_of(camera).to_np()
    for annotation, grasp in zip(annotations, perceived):
        np.testing.assert_allclose(
            box.global_transform.to_np() @ grasp.grasp_pose.to_np(),
            world_T_camera
            @ get_transform_matrix_from_q(annotation.rotation, annotation.translation),
            atol=1e-6,
        )


# %% picking an object up with a predicted grasp


def test_a_can_is_picked_up_with_a_grasp_predicted_from_what_the_camera_sees(
    can_on_the_table, tracy_gripper, annotator_output_buffer
):
    """
    A pick-up asking the can for its grasps takes one of those predicted on what Tracy's
    camera sees of it, and ends with the can hanging off the gripper.
    """
    world, robot, extensions, can = can_on_the_table
    annotate_what_the_camera_sees_of(can, robot.get_default_camera(), tracy_gripper)
    graspable = graspable_annotation(world, can)
    pick = StatementNode(
        statement=a(PickUpAction)(
            grasp_candidate=variable(
                GraspCandidate, domain=graspable.grasp_candidates()
            ),
            arm=robot.left_arm,
        )
    )

    run_plan(Sequence(nodes=[ParkArmsAction(arms=robot.all_arms), pick]), extensions)

    assert pick.grounded_nodes[-1].grasp_candidate.grasp_pose in [
        grasp.grasp_pose for grasp in graspable.perceived_grasps
    ]
    assert world.get_connection(robot.left_arm.end_effector.tool_frame, can) is not None
