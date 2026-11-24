from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner, FailsOnObservingFalse
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from giskardpy.motion_statechart.ros_context import RosContextExtension
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from rclpy.action import ActionClient
from rclpy.node import Node
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.reasoning.robot_predicates import VisibleTo
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
    Vector3,
)
from krrood_spatial_math.spatial_types import Pose, Point3
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cloth,
    LaundryBasket,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from typing_extensions import Any, Self, TYPE_CHECKING

from coraplex.datastructures.enums import PerceptionSource
from coraplex.plans.context_extensions import RobotAccess
from coraplex.exceptions import (
    AmbiguousDetection,
    CameraImageAxisVertical,
    PerceivedObjectNotInWorld,
    PerceptionSourceUnavailable,
    UnidentifiedDetections,
)

if TYPE_CHECKING:
    from robokudo_msgs.msg import ObjectDesignator

ROBOKUDO_QUERY_ACTION_NAME = "robokudo/query"
"""
Name of the action RoboKudo answers perception queries on.
"""

# %% queries


@dataclass
class PerceptionQuery(SubclassJSONSerializer):
    """
    What to look for and where.

    Travels to whichever process answers it, so the world and the robot are carried by
    reference and resolved against the world on the receiving side.
    """

    semantic_annotation: type[SemanticAnnotation]
    """
    The semantic annotation for which to perceive.
    """

    region: VolumetricBoundingBox
    """
    The region in which the object should be detected.
    """

    robot: AbstractRobot
    """
    ' Robot annotation of the robot that should perceive the object.
    """

    world: World
    """
    The world in which the object should be detected.
    """

    trust_detected_orientation: bool = True
    """
    Whether to also apply the perception source's detected orientation.

    When False, only the detected position is applied to the world; the object's
    existing orientation is kept. See :meth:`Detection.apply_to`.
    """

    def from_world(self) -> list[Body]:
        """
        Answer this query from the world model alone.

        :return: The bodies of the queried annotation that lie inside the region and are
            visible to the robot's camera. A body several matching annotations describe
            is one object, so it is reported once.
        """
        bodies = dict.fromkeys(
            body
            for semantic_annotation in self.world.get_semantic_annotations_by_type(
                self.semantic_annotation
            )
            for body in semantic_annotation.bodies
        )
        region_bodies = [
            body
            for body in bodies
            if self.region.contains(body.global_transform.position)
        ]

        robot_camera = self.robot.get_default_camera()
        return [
            body
            for body in region_bodies
            if VisibleTo(entity=body, camera=robot_camera)()
        ]

    def to_json(self, **kwargs) -> dict[str, Any]:
        result = super().to_json(**kwargs)
        result["semantic_annotation"] = to_json(self.semantic_annotation, **kwargs)
        result["region"] = to_json(self.region, **kwargs)
        result["robot_id"] = to_json(self.robot.id, **kwargs)
        result["trust_detected_orientation"] = self.trust_detected_orientation
        return result

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(from_json_kwargs=kwargs)
        return cls(
            semantic_annotation=from_json(data["semantic_annotation"], **kwargs),
            region=from_json(data["region"], **kwargs),
            robot=tracker.get(key=from_json(data["robot_id"])),
            world=kwargs["world"],
            trust_detected_orientation=data.get("trust_detected_orientation", True),
        )


# %% detections


@dataclass
class Detection:
    """
    One object reported by a perception source.
    """

    semantic_annotation: type[SemanticAnnotation]
    """
    The annotation the perception source identified the object as.
    """

    pose: Pose
    """
    Where the perception source saw the object.
    """

    def apply_to(
        self, world: World, trust_orientation: bool = True
    ) -> list[IsPerceivable]:
        """
        Write this detection into ``world``, moving the perceived body to where it was
        seen.

        :param world: The world holding the annotations this detection refers to.
        :param trust_orientation: Whether to also apply the detected orientation. When
            False, only the detected position is applied and the body keeps its existing
            orientation; useful while a perception source's orientation estimate is not
            yet reliable enough for grasp planning.
        :return: The annotations that were updated.
        :raises PerceivedObjectNotInWorld: If the world holds no such annotation.
        :raises AmbiguousDetection: If the annotation describes more than one body.
        """
        matching_annotations = self.resolve_annotations(world)
        matching_annotation = matching_annotations[0]
        body = matching_annotation.root
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]

        # Foundation Pose may return a upside down pose. this detects and fixes it by rotation around x by 180 degree
        detected_global_pose = world.transform(self.pose, world.root).homogeneous_matrix
        detected_global_pose = self._with_z_up_and_x_towards(
            root_T_frame=detected_global_pose,
            root_P_target=robot.root.global_transform.position,
        )

        if trust_orientation:
            body.parent_connection.origin = detected_global_pose
        else:
            parent_T_object = world.transform(
                detected_global_pose, body.parent_connection.parent
            )
            parent_origin = body.parent_connection.origin
            body.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=parent_T_object.position,
                    rotation_matrix=parent_origin.rotation_matrix,
                    reference_frame=body.parent_connection.parent,
                )
            )
        return matching_annotations

    @staticmethod
    def _with_z_up_and_x_towards(
        root_T_frame: HomogeneousTransformationMatrix, root_P_target: Point3
    ) -> HomogeneousTransformationMatrix:
        """
        Return ``root_T_frame`` rotated by 180° around its own axes so that its z axis
        points along the world z axis and its x axis points towards ``root_P_target`` as
        closely as those rotations allow.

        The position is left untouched, and every axis stays on the axis it started on.
        :param root_T_frame:``HomogeneousTransformationMatrix`` which will be rotated
        :param root_P_target:``Point3`` towards which root_T_frame x axis will point
        :return: rotated transformation matrix
        """
        root_V_target = root_P_target - root_T_frame.position

        flip_z = root_T_frame[2, 2] < 0
        flip_x = root_T_frame.rotation_matrix.x_vector().dot(root_V_target) > 0

        return root_T_frame @ HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=np.pi if flip_z and not flip_x else 0,
            pitch=np.pi if flip_z and flip_x else 0,
            yaw=np.pi if flip_x and not flip_z else 0,
        )

    def resolve_annotations(self, world: World) -> list[IsPerceivable]:
        """
        Find the annotations in ``world`` that this detection identified.

        Several annotations may describe the same body, which is not ambiguous; only an
        annotation describing more than one body is.

        :param world: The world to search.
        :return: The matching annotations, all sharing one root body.
        :raises PerceivedObjectNotInWorld: If the world holds no instance of the
            annotation.
        :raises AmbiguousDetection: If the matching annotations sit on different bodies.
        """
        candidates = world.get_semantic_annotations_by_type(self.semantic_annotation)
        if not candidates:
            raise PerceivedObjectNotInWorld(self.semantic_annotation)
        bodies = {annotation.root for annotation in candidates}
        if len(bodies) > 1:
            raise AmbiguousDetection(self.semantic_annotation, len(bodies))
        return candidates


# %% perception sources


class PerceptionInterface(ABC):
    """
    A source of detections.
    """

    @abstractmethod
    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        """
        Report every object this source sees for a perception query.

        :param query: What to look for and where.
        :return: One detection per object seen, empty when the source saw nothing.
        """

    def detect(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> Detection | None:
        """
        Answer a perception query with the one object it asked about.

        :param query: What to look for and where.
        :param accept_first_if_multiple: Whether several candidates may be resolved by
            taking the first one. When False, several candidates raise
            :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen
            between.
        :return: The object this source saw, None when it saw nothing.
        :raises UnidentifiedDetections: If several candidates were seen and none of them
            may be picked.
        """
        detections = self.look_for(query)
        if not detections:
            return None
        if len(detections) > 1 and not accept_first_if_multiple:
            raise UnidentifiedDetections(query.semantic_annotation, len(detections))
        return detections[0]

    @staticmethod
    def for_source(
        source: PerceptionSource, ros_node: Node | None = None
    ) -> PerceptionInterface:
        """
        :param source: The kind of source that answers the queries.
        :param ros_node: Node a RoboKudo pipeline is reached through.
        :return: The source to answer queries with.
        """
        match source:
            case PerceptionSource.WORLD_MODEL:
                return WorldPerception()
            case PerceptionSource.ROBOKUDO:
                return RoboKudoPerception(ros_node=ros_node)


@dataclass
class WorldPerception(PerceptionInterface):
    """
    Source that reads the objects straight out of the world model.

    Reports what a perfect sensor would see: every body of the queried annotation that
    lies inside the region and is visible to the robot's camera, at the pose the world
    already holds.
    """

    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        annotation_by_body = {}
        for annotation in query.world.get_semantic_annotations_by_type(
            query.semantic_annotation
        ):
            annotation_by_body.setdefault(annotation.root, type(annotation))

        return [
            Detection(
                semantic_annotation=annotation_by_body[body], pose=body.global_pose
            )
            for body in query.from_world()
            if body in annotation_by_body
        ]


@dataclass
class RoboKudoPerception(PerceptionInterface):
    """
    Source that asks a running RoboKudo pipeline over its query action.
    """

    ros_node: Node
    """
    Node the action client is created on.
    """

    action_name: str = ROBOKUDO_QUERY_ACTION_NAME
    """
    Action RoboKudo answers queries on.
    """

    server_timeout: timedelta = field(default=timedelta(seconds=10))
    """
    How long to wait for the action server before giving up.
    """

    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        # RoboKudo's messages only exist where its pipeline is installed, so they are
        # imported here rather than at module level: reading the world in simulation must
        # not depend on them.
        from robokudo_msgs.action import Query
        from robokudo_msgs.msg import ObjectDesignator

        client = ActionClient(self.ros_node, Query, self.action_name)
        if not client.wait_for_server(timeout_sec=self.server_timeout.total_seconds()):
            raise PerceptionSourceUnavailable(self.action_name)

        goal = Query.Goal(
            obj=ObjectDesignator(type=query.semantic_annotation.__name__.lower())
        )
        result = client.send_goal(goal).result
        return [
            self._to_detection(designator, query)
            for designator in result.res
            if designator.pose and self._can_be_requested_object(designator, query)
        ]

    @staticmethod
    def _can_be_requested_object(
        designator: ObjectDesignator, query: PerceptionQuery
    ) -> bool:
        """
        Whether a reported designator could be the object the query asked for.

        A pipeline that only localizes reports no class label, and every object it found
        is then a candidate. One that classifies lets the objects that were not asked
        for be discarded here.

        :param designator: The designator the pipeline reported.
        :param query: The query it was answering.
        :return: True if the designator is a candidate for the requested object.
        """
        if not designator.type:
            return True
        return (
            designator.type.strip().lower()
            == query.semantic_annotation.__name__.lower()
        )

    def _to_detection(
        self, designator: ObjectDesignator, query: PerceptionQuery
    ) -> Detection:
        """
        Convert one RoboKudo object designator into a detection.

        The detection is identified as the queried annotation: a pipeline that classifies
        has already been narrowed to it by :meth:`_can_be_requested_object`, and one that
        only localizes answered where the queried object is.

        :param designator: The designator RoboKudo reported.
        :param query: The query it was answering, supplying the annotation and the frame.
        :return: The detection carrying the queried annotation and the reported pose.
        """
        world = query.world
        pose_stamped = designator.pose[0]
        return Detection(
            semantic_annotation=query.semantic_annotation,
            pose=Pose.from_xyz_quaternion(
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
                reference_frame=world.get_kinematic_structure_entity_by_name(
                    pose_stamped.header.frame_id
                ),
            ),
        )


# %% perceiving inside the motion chart


@dataclass(eq=False, repr=False, kw_only=True)
class PerceptionTask(EndedByOwner, FailsOnObservingFalse, StatechartNode):
    """
    Statechart node that answers a perception query and writes what it saw into the
    world.

    The node moves nothing, so it adds no motion constraints and its owner decides when
    it succeeded, like a task of a motion. The query is answered on the node's first tick,
    so the whole detection takes one tick however long the source needs to reply, and the
    node then observes ``TRUE`` so the surrounding sequence continues. A source that saw
    nothing makes it observe ``FALSE``, which fails it, so a surrounding
    :class:`~cramph.composites.TryInOrder` can go and look somewhere else.

    .. warning:: That tick blocks until the source replies, which on the real robot holds
        up the control loop for as long as the pipeline takes to answer.
    """

    query: PerceptionQuery
    """
    What to look for and where.
    """

    answered_by: PerceptionSource
    """
    Which kind of source answers the query.

    Carried by the node rather than read from the context, because on the real robot the
    chart is answered in the controller's process, whose context does not say how the
    plan is executed.
    """

    perception_source: PerceptionInterface | None = field(init=False, default=None)
    """
    The source answering the query, resolved during :meth:`set_up`.
    """

    _observation: ObservationStateValues | None = field(
        init=False, default=None, repr=False
    )
    """
    The observation this node keeps once the query has been answered, None before that.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    def set_up(self) -> None:
        super().set_up()
        self.perception_source = PerceptionInterface.for_source(
            self.answered_by,
            self.context.require_extension(extension_type=RosContextExtension).ros_node,
        )

    def on_start(self) -> None:
        self._observation = None

    def on_tick(self) -> ObservationStateValues | None:
        if self._observation is None:
            self._observation = self._answer_query()
        return self._observation

    def _answer_query(self) -> ObservationStateValues:
        """
        Answer the query and write what was seen into the world.

        :return: True once the object was seen, False when the source saw nothing.
        """
        detection = self.perception_source.detect(
            self.query, self.accept_first_if_multiple
        )
        if detection is None:
            return ObservationStateValues.FALSE
        detection.apply_to(
            self.query.world, trust_orientation=self.query.trust_detected_orientation
        )
        return ObservationStateValues.TRUE


# %% taking hold of the laundry in a basket


@dataclass(eq=False, repr=False, kw_only=True)
class LaundryGraspDetection(EndedByOwner, FailsOnObservingFalse, StatechartNode):
    """
    Statechart node that records on a basket where to take hold of its laundry, as
    RoboKudo's cloth pickup engine does, but read from the world model through the
    robot's default camera: on the highest cloth it sees in the basket, approaching
    straight down and closing along the image's horizontal axis laid flat.
    """

    basket: LaundryBasket
    """
    The basket whose laundry to take hold of.
    """

    _observation: ObservationStateValues | None = field(
        init=False, default=None, repr=False
    )
    """
    The observation this node keeps once it looked into the basket, None before that.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RobotAccess,)

    def on_start(self) -> None:
        self._observation = None

    def on_tick(self) -> ObservationStateValues | None:
        if self._observation is None:
            self._observation = self._record_laundry_grasp()
        return self._observation

    def _record_laundry_grasp(self) -> ObservationStateValues:
        """
        Record on the basket the grasp on the highest point of the laundry the camera
        sees in it.

        :return: True once the grasp was recorded, False when the camera sees no cloth
            in the basket.
        """
        robot = self.context.require_extension(extension_type=RobotAccess).robot
        camera = robot.get_default_camera()
        world_P_laundry = self.visible_laundry_points(robot=robot, camera=camera)
        if len(world_P_laundry) == 0:
            return ObservationStateValues.FALSE
        highest_point = world_P_laundry[np.argmax(world_P_laundry[:, 2])]
        grasp_pose = self.top_down_grasp_pose(grasp_point=highest_point, camera=camera)
        with self.context.world.modify_world():
            self.basket.update_laundry_grasp_pose(grasp_pose)
        return ObservationStateValues.TRUE

    def visible_laundry_points(
        self, *, robot: AbstractRobot, camera: Camera
    ) -> np.ndarray:
        """
        :param robot: The robot carrying the camera.
        :param camera: The camera looking into the basket.
        :return: The points of cloth inside the basket's bounding box that the camera
            sees, in the world frame, one per row.
        """
        world_P_seen, seen_bodies = self.first_hits_past_robot(
            robot=robot, camera=camera
        )
        cloth_bodies = {
            cloth.root
            for cloth in self.context.world.get_semantic_annotations_by_type(Cloth)
        }
        on_cloth = np.array([body in cloth_bodies for body in seen_bodies], dtype=bool)
        world_P_cloth = world_P_seen[on_cloth]
        return world_P_cloth[self.are_inside_basket(world_P_points=world_P_cloth)]

    def first_hits_past_robot(
        self, *, robot: AbstractRobot, camera: Camera
    ) -> tuple[np.ndarray, list[Body]]:
        """
        :param robot: The robot whose own bodies the camera sees past, as the camera
            sits inside its housing.
        :param camera: The camera to cast the pixel rays of.
        :return: The point each pixel ray hits first outside the robot, in the world
            frame, one per row, and the body hit there, for each ray that hits any.
        """
        camera_model = PinholeCameraModel.from_field_of_view(
            resolution=camera.resolution, field_of_view=camera.field_of_view
        )
        ray_tracer = RayTracer(self.context.world)
        rays = ray_tracer.create_camera_rays(
            camera.world_T_forward_view, camera_model=camera_model
        )
        hits = ray_tracer.ray_test(
            rays.origin_points,
            rays.origin_points + rays.direction_vectors,
            multiple_hits=True,
            min_distance=camera.camera_range.minimum_distance,
            max_distance=camera.camera_range.maximum_distance,
        )
        robot_bodies = set(robot.bodies)
        outside_robot = np.array(
            [body not in robot_bodies for body in hits.bodies], dtype=bool
        )
        ray_indices = hits.ray_indices[outside_robot]
        hit_points = hits.hit_points[outside_robot]
        hit_bodies = [body for body, kept in zip(hits.bodies, outside_robot) if kept]
        distances = np.linalg.norm(hit_points - rays.origin_points[ray_indices], axis=1)
        by_ray_then_distance = np.lexsort((distances, ray_indices))
        _, first_of_each_ray = np.unique(
            ray_indices[by_ray_then_distance], return_index=True
        )
        first_hits = by_ray_then_distance[first_of_each_ray]
        return hit_points[first_hits], [hit_bodies[hit] for hit in first_hits]

    def are_inside_basket(self, *, world_P_points: np.ndarray) -> np.ndarray:
        """
        :param world_P_points: Points in the world frame, one per row.
        :return: Whether each point lies inside the basket's bounding box.
        """
        basket = self.basket.root
        basket_box = basket.collision.as_bounding_box_collection_in_frame(
            basket
        ).bounding_box()
        basket_T_world = self.context.world.compute_forward_kinematics_np(
            basket, self.context.world.root
        )
        basket_P_points = (
            world_P_points @ basket_T_world[:3, :3].T + basket_T_world[:3, 3]
        )
        lower = np.array([basket_box.min_x, basket_box.min_y, basket_box.min_z])
        upper = np.array([basket_box.max_x, basket_box.max_y, basket_box.max_z])
        return np.all((basket_P_points >= lower) & (basket_P_points <= upper), axis=1)

    def top_down_grasp_pose(self, *, grasp_point: np.ndarray, camera: Camera) -> Pose:
        """
        :param grasp_point: Where to take hold, in the world frame.
        :param camera: The camera whose image decides the closing direction.
        :return: A grasp at the point approaching straight down along its x-axis and
            closing along its y-axis, the image's horizontal axis laid flat.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down.
        """
        world = self.context.world
        image_x_axis = camera.root.global_transform.to_np()[:3, 0]
        flat_image_x_axis = np.array([image_x_axis[0], image_x_axis[1], 0.0])
        length = np.linalg.norm(flat_image_x_axis)
        if np.isclose(length, 0.0):
            raise CameraImageAxisVertical(camera_name=camera.root.name)
        closing = flat_image_x_axis / length
        rotation = RotationMatrix.from_vectors(
            x=Vector3(0.0, 0.0, -1.0, reference_frame=world.root),
            y=Vector3(*closing, reference_frame=world.root),
            reference_frame=world.root,
        )
        return Pose(
            position=Point3(*grasp_point, reference_frame=world.root),
            orientation=rotation.quaternion,
            reference_frame=world.root,
        )
