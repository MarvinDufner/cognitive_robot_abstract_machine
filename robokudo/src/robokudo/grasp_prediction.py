"""
Grasps predicted on the point cloud of a single object by GraspGenX, a diffusion network
conditioned on the gripper, so that one model serves every two-finger gripper.

.. note:: GraspGenX is installed without its dependencies, since it pins versions of
    numpy and trimesh the workspace has outgrown:
    ``pip install --no-deps git+https://github.com/NVlabs/GraspGenX@16349a5c5d5cea1da6c3b0541d60c8312f0b0db6``.
    Its weights are downloaded from Hugging Face on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import omegaconf
import torch
import trimesh
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.x_grippers import make_sweep_volume_gripper_info
from huggingface_hub import hf_hub_download
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from typing_extensions import Self

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.grasping.grasp_candidates import PerceivedGrasp
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.world_description.world_entity import Body

# %% checkpoint


class GraspGenXCheckpointFile(StrEnum):
    """
    The files of the released GraspGenX checkpoint in its Hugging Face repository.
    """

    GENERATOR_CONFIGURATION = "release/gen/config.yaml"
    GENERATOR_WEIGHTS = "release/gen/epoch_736.pth"
    DISCRIMINATOR_CONFIGURATION = "release/dis/config.yaml"
    DISCRIMINATOR_WEIGHTS = "release/dis/epoch_1056.pth"


# %% the gripper as the network sees it


@dataclass(kw_only=True)
class FingerSpace:
    """
    The box between the fingertips of a gripper standing in one state, in its hand
    frame.
    """

    center: np.ndarray
    """
    The centre of the box, in metres.
    """

    extents: np.ndarray
    """
    The size of the box along the closing axis, across the fingers and along the
    approach axis, in metres.
    """


@dataclass(kw_only=True)
class GraspGenXGripper:
    """
    A two-finger gripper as GraspGenX is conditioned on it.

    Its hand frame sits at the origin of the gripper's root body, with the z-axis along
    the approach axis and the x-axis along the closing axis.
    """

    open_finger_space: FingerSpace
    """
    The space between the fingertips of the open gripper.
    """

    half_open_finger_space: FingerSpace
    """
    The space between the fingertips of the gripper standing half as wide open.
    """

    hand_T_grasp: np.ndarray
    """
    The grasp frame, as
    :class:`~semantic_digital_twin.grasping.grasp_candidates.GraspCandidate` describes
    it, in the hand frame: at the point the closed fingertips meet.
    """

    surface_points: np.ndarray
    """
    Points on the collision surface of the open gripper, one per row, in the hand frame.
    """

    @classmethod
    def from_two_finger_gripper(
        cls, *, gripper: HasTwoFingers, number_of_surface_points: int = 2000
    ) -> Self:
        """
        :param gripper: A gripper of a robot in its world.
        :param number_of_surface_points: How many points of its surface are sampled.
        :return: The gripper measured off its description.
        """
        approach = gripper.approach_axis.to_np()[:3]
        closing = gripper.closing_axis.to_np()[:3]
        approach, closing = (
            approach / np.linalg.norm(approach),
            closing / np.linalg.norm(closing),
        )
        hand_T_tool = np.eye(4)
        hand_T_tool[:3, :3] = np.stack([closing, np.cross(approach, closing), approach])
        tool_P_hand = gripper._world.compute_forward_kinematics_np(
            gripper.tool_frame, gripper.root
        )[:3, 3]
        hand_T_tool[:3, 3] = -hand_T_tool[:3, :3] @ tool_P_hand

        open_state = gripper.get_joint_state_by_type(GripperState.OPEN)
        closed_state = gripper.get_joint_state_by_type(GripperState.CLOSE)

        def state_closed_by(fraction: float) -> JointState:
            return JointState.from_mapping(
                {
                    connection: open_value
                    + fraction * (closed_state.target_value(connection) - open_value)
                    for connection, open_value in open_state.items()
                }
            )

        half_width = gripper.separation_when_open / 2
        half_open_state = state_closed_by(
            brentq(
                lambda fraction: (
                    gripper.separation_at(state_closed_by(fraction)) - half_width
                ),
                0.0,
                1.0,
            )
        )
        # The grasp frame's x-, y- and z-axes are the hand frame's z-, x- and y-axes.
        hand_T_grasp = np.eye(4)
        hand_T_grasp[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        hand_T_grasp[:3, 3] = (
            hand_T_tool @ gripper.tool_P_grasp_when_closed.to_np().reshape(-1)
        )[:3]
        return cls(
            open_finger_space=cls._finger_space(
                gripper=gripper, joint_state=open_state, hand_T_tool=hand_T_tool
            ),
            half_open_finger_space=cls._finger_space(
                gripper=gripper, joint_state=half_open_state, hand_T_tool=hand_T_tool
            ),
            hand_T_grasp=hand_T_grasp,
            surface_points=cls._surface_points(
                gripper=gripper,
                joint_state=open_state,
                hand_T_tool=hand_T_tool,
                number_of_points=number_of_surface_points,
            ),
        )

    @staticmethod
    def _surface_points(
        *,
        gripper: HasTwoFingers,
        joint_state: JointState,
        hand_T_tool: np.ndarray,
        number_of_points: int,
    ) -> np.ndarray:
        """
        :param gripper: The gripper to sample.
        :param joint_state: The state to stand it in.
        :param hand_T_tool: The tool frame in the hand frame.
        :param number_of_points: How many points to sample.
        :return: Points evenly spread over the collision surface of every body of the
            gripper standing in that state, in the hand frame.
        """
        world = gripper._world
        with world.reset_state_context():
            joint_state.apply_to(world)
            meshes = [
                body.collision.combined_mesh.apply_transform(
                    hand_T_tool
                    @ world.compute_forward_kinematics_np(gripper.tool_frame, body)
                )
                for body in world.get_kinematic_structure_entities_of_branch(
                    gripper.root
                )
                if isinstance(body, Body) and body.has_collision()
            ]
        points, _ = trimesh.sample.sample_surface(
            trimesh.util.concatenate(meshes), number_of_points, seed=0
        )
        return np.asarray(points)

    @staticmethod
    def _finger_space(
        *, gripper: HasTwoFingers, joint_state: JointState, hand_T_tool: np.ndarray
    ) -> FingerSpace:
        """
        :param gripper: The gripper to measure.
        :param joint_state: The state to stand it in.
        :param hand_T_tool: The tool frame in the hand frame.
        :return: The box between the fingertips in that state, as wide as they stand
            apart and as long and wide as a fingertip, in the hand frame.
        """
        with gripper._world.reset_state_context():
            joint_state.apply_to(gripper._world)
            fingertips = gripper.fingertips
            tool_P_center = fingertips.tool_P_grasp.to_np().reshape(-1)
            separation = fingertips.separation
            tip_bounds = [
                tip.collision.as_bounding_box_collection_in_frame(gripper.tool_frame)
                .bounding_box()
                .to_array_bounds()
                for tip in fingertips.both
            ]
            tool_extents = np.mean(
                [bounds.upper - bounds.lower for bounds in tip_bounds], axis=0
            )
        extents = np.abs(hand_T_tool[:3, :3]) @ tool_extents
        extents[0] = separation
        return FingerSpace(center=(hand_T_tool @ tool_P_center)[:3], extents=extents)


# %% predicted grasps


@dataclass(kw_only=True)
class PredictedGrasp:
    """
    A grasp the network predicted on a point cloud.
    """

    points_T_grasp: np.ndarray
    """
    The grasp frame, as
    :class:`~semantic_digital_twin.grasping.grasp_candidates.GraspCandidate` describes
    it, in the frame of the points it was predicted on.
    """

    confidence: float
    """
    How likely the network rates the grasp to succeed, from 0 to 1.
    """

    def to_perceived_grasp(
        self, *, body: Body, body_T_points: np.ndarray
    ) -> PerceivedGrasp:
        """
        :param body: The root body of the object grasped.
        :param body_T_points: The frame of the points, in the frame of ``body``.
        :return: This grasp relative to ``body``.
        """
        return PerceivedGrasp(
            grasp_pose=HomogeneousTransformationMatrix(
                body_T_points @ self.points_T_grasp, reference_frame=body
            ).pose,
            confidence=self.confidence,
        )


# %% predicting grasps


@dataclass(kw_only=True)
class GraspPosePredictor:
    """
    Predicts the grasps of a two-finger gripper on point clouds of single objects.
    """

    gripper: GraspGenXGripper
    """
    The gripper to predict the grasps of.
    """

    number_of_samples: int = 100
    """
    How many grasps the network samples per point cloud before they are ranked.
    """

    model_repository: str = "adithyamurali/GraspGenXModel"
    """
    The Hugging Face repository holding the network's weights.
    """

    model_revision: str = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
    """
    The revision of :attr:`model_repository` to load.
    """

    collision_clearance: float = 0.01
    """
    How close the gripper may come to the scene around the object, in metres.
    """

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    """
    The torch device the network runs on.
    """

    _sampler: GraspGenXSampler = field(init=False, repr=False)
    """
    The network, conditioned on :attr:`gripper`.
    """

    def __post_init__(self):
        configuration, discriminator_configuration = (
            omegaconf.OmegaConf.load(self._download(checkpoint_file=checkpoint_file))
            for checkpoint_file in (
                GraspGenXCheckpointFile.GENERATOR_CONFIGURATION,
                GraspGenXCheckpointFile.DISCRIMINATOR_CONFIGURATION,
            )
        )
        configuration.discriminator = discriminator_configuration.discriminator
        if self.device == "cpu":
            # The attention's fast path runs on CUDA only; the weights are the same.
            configuration.diffusion.ptv3vanilla.enable_flash = False
            configuration.discriminator.ptv3vanilla.enable_flash = False
        model = GraspGen.from_config(
            configuration.diffusion, configuration.discriminator
        )
        model.load_state_dict(
            self._download(checkpoint_file=GraspGenXCheckpointFile.GENERATOR_WEIGHTS),
            self._download(
                checkpoint_file=GraspGenXCheckpointFile.DISCRIMINATOR_WEIGHTS
            ),
        )
        self._sampler = GraspGenXSampler(
            configuration,
            gripper_info=make_sweep_volume_gripper_info(
                self.gripper.open_finger_space.extents,
                self.gripper.open_finger_space.center,
                self.gripper.half_open_finger_space.extents,
                self.gripper.half_open_finger_space.center,
            ),
            model=model.to(self.device).eval(),
        )

    def _download(self, *, checkpoint_file: GraspGenXCheckpointFile) -> str:
        """
        :param checkpoint_file: A file of the checkpoint.
        :return: Its local path, downloaded unless already cached.
        """
        return hf_hub_download(
            repo_id=self.model_repository,
            filename=checkpoint_file,
            revision=self.model_revision,
        )

    def predict(
        self, *, points: np.ndarray, scene_points: np.ndarray | None = None
    ) -> list[PredictedGrasp]:
        """
        :param points: The points of one object, one per row, in any frame.
        :param scene_points: The points around the object, its own included, in the same
            frame; grasps bringing the gripper within :attr:`collision_clearance` of
            those not on the object are left out.
        :return: The predicted grasps in the frame of ``points``, the most confident
            first.
        """
        hand_poses, confidences, _ = self._sampler.sample(
            torch.as_tensor(points, dtype=torch.float32, device=self.device),
            num_grasps=self.number_of_samples,
        )
        if len(hand_poses) == 0:
            return []
        hand_poses, confidences = hand_poses.cpu().numpy(), confidences.cpu().numpy()
        hand_poses[:, 3] = [0, 0, 0, 1]
        if scene_points is not None:
            clear = self._clear_of(
                obstacle_points=self._obstacles_around(
                    points=points, scene_points=scene_points
                ),
                hand_poses=hand_poses,
            )
            hand_poses, confidences = hand_poses[clear], confidences[clear]
        return [
            PredictedGrasp(
                points_T_grasp=hand_poses[index] @ self.gripper.hand_T_grasp,
                confidence=float(confidences[index]),
            )
            for index in np.argsort(-confidences)
        ]

    def _obstacles_around(
        self, *, points: np.ndarray, scene_points: np.ndarray
    ) -> np.ndarray:
        """
        :param points: The points of the object.
        :param scene_points: The points around it, its own included.
        :return: The scene points farther than :attr:`collision_clearance` from the
            object.
        """
        distances, _ = cKDTree(points).query(scene_points)
        return scene_points[distances > self.collision_clearance]

    def _clear_of(
        self, *, obstacle_points: np.ndarray, hand_poses: np.ndarray
    ) -> np.ndarray:
        """
        :param obstacle_points: The points the gripper must keep clear of.
        :param hand_poses: The poses of the hand frame, one per grasp.
        :return: For each pose, whether the gripper's surface keeps farther than
            :attr:`collision_clearance` from every obstacle point.
        """
        if len(obstacle_points) == 0:
            return np.ones(len(hand_poses), dtype=bool)
        surfaces = (
            np.einsum("kij,mj->kmi", hand_poses[:, :3, :3], self.gripper.surface_points)
            + hand_poses[:, None, :3, 3]
        )
        distances, _ = cKDTree(obstacle_points).query(
            surfaces.reshape(-1, 3), distance_upper_bound=self.collision_clearance
        )
        return np.isinf(distances).reshape(len(hand_poses), -1).all(axis=1)
