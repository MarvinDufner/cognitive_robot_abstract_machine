from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import Any, List, Optional, Tuple, Union

from semantic_digital_twin.spatial_types import Vector3

from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import (
    Arms,
    CuttingTechnique,
    MixingPattern,
    MovementType,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)
from coraplex.exceptions import (
    MissingWaypoints,
    MotionDidNotFinish,
    WipingTargetMissing,
)
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import HasTcpGoalThresholds
from coraplex.view_manager import ViewManager
from coraplex.robot_plans.actions.composite.tool_paths import (
    ToolPath,
    ToolPathSegment,
    build_container_path,
    build_cutting_path,
    build_surface_path,
    planar_raster_xy,
    planar_spiral_xy,
    planar_sweep_x,
)
from coraplex.robot_plans.motions.gripper import (
    MoveTCPWaypointsAlignedMotion,
    MoveToolCenterPointMotion,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FullBodyControlledAction(ActionDescription, ABC):
    """
    An action that executes its plan with the robot's full body controlled, so the base
    can support the arm motion.
    """

    def execute(self) -> Any:
        """
        Perform the action's plan with full body control enabled, restoring the previous
        control state afterwards.
        """
        previous_full_body_controlled = self._enable_full_body_control()
        try:
            self._perform_plan()
        finally:
            self._restore_full_body_control(previous_full_body_controlled)

    def _perform_plan(self) -> None:
        """
        Add the action plan as a subplan and perform it.
        """
        self.add_subplan(self.action_plan).perform()

    def _enable_full_body_control(self) -> Optional[bool]:
        """
        :return: The previous full-body-control state, or None if the robot has no
            mobile base.
        """
        if not isinstance(self.robot, HasMobileBase):
            return None
        previous = self.robot.mobile_base.full_body_controlled
        self.robot.mobile_base.full_body_controlled = True
        return previous

    def _restore_full_body_control(self, previous: Optional[bool]) -> None:
        """
        :param previous: The full-body-control state to restore, or None if the robot
            has no mobile base.
        """
        if previous is None:
            return
        self.robot.mobile_base.full_body_controlled = previous


@dataclass(kw_only=True)
class ToolMotionAction(FullBodyControlledAction, ABC, HasTcpGoalThresholds):
    """
    An action that moves a tool along a sampled tool path while keeping the tool aligned
    with its target.
    """

    arm: Arms
    """
    The arm holding the tool.
    """

    tool: Tool
    """
    The tool that performs the motion.
    """

    pointer_stride: int = 1
    """
    Keep every Nth sampled waypoint for execution.
    """

    standing_pose: Optional[Pose] = None
    """
    Where the robot will stand while this motion runs.

    The waypoints are worked out when the plan is built, so a plan that navigates first
    would otherwise measure reach from wherever the robot happened to start. Only used
    with :attr:`skip_unreachable`.
    """

    skip_unreachable: bool = False
    """
    Drop the waypoints the arm cannot reach from where the robot stands.

    The tool follows the waypoints in order and cannot skip ahead, so a path that starts
    out of reach never starts at all. Dropping those waypoints covers what this placement
    can cover and leaves the rest to the next one.

    .. note:: The bound is purely kinematic, so a kept waypoint is reachable at best; it
        may still be blocked or need a posture the arm cannot hold.
    """

    @abstractmethod
    def _build_tool_path(self) -> ToolPath:
        """
        :return: The tool path of this action in its local frame.
        """

    @abstractmethod
    def _path_frame(self) -> HomogeneousTransformationMatrix:
        """
        :return: The frame the tool path is expressed in.
        """

    @property
    @abstractmethod
    def _alignment_target(self) -> Optional[Union[Body, Pose]]:
        """
        :return: The target the tool is aligned with during the motion.
        """

    @cached_property
    def _waypoints(self) -> List[Point3]:
        """
        :return: The sampled waypoints of the tool path in the world frame.
        """
        _, points, _ = self._build_tool_path().sample(frame=self._path_frame())
        stride = max(1, int(self.pointer_stride))
        if self.skip_unreachable:
            points = points[self._within_reach(points)]
        waypoints = [
            Point3(x=point[0], y=point[1], z=point[2], reference_frame=self.world.root)
            for point in points
        ][::stride]
        if not waypoints:
            raise MissingWaypoints(self)
        return waypoints

    def _within_reach(self, points: np.ndarray) -> np.ndarray:
        """
        :param points: Sampled path points in the world frame.
        :return: Which of them the arm can reach from where the robot stands, measured
            horizontally, as the arm's own reach is.
        """
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        if self.standing_pose is None:
            standing = self.world.compute_forward_kinematics_np(
                self.world.root, self.robot.root
            )[:2, 3]
        else:
            standing = self.standing_pose.to_np()[:2, 3]
        distances = np.linalg.norm(points[:, :2] - standing, axis=1)
        return distances <= arm.maximum_reach()

    @property
    def _alignment_pairs(self) -> List[AlignmentPair]:
        """
        :return: The normal pairs that keep the tool aligned with its target during
            the motion, or an empty list if there is no alignment target.
        """
        target = self._alignment_target
        if target is None:
            return []
        return self.tool.tool_alignment(target)

    admittance_mass: Optional[Vector3] = None
    """
    Virtual mass the press yields with, per axis, in kg.

    ``None`` leaves the default. Only meaningful together with :attr:`desired_force`.
    """
    admittance_damping: Optional[Vector3] = None
    """
    Virtual damping the press yields with, per axis, in N s/m.

    ``None`` leaves the default. Only meaningful together with :attr:`desired_force`.
    """
    desired_force: Optional[Vector3] = None
    """
    Contact force to hold against the target while the tool moves, in the world frame.

    ``None`` follows the tool path as generated, which keeps the tool just clear of the
    surface. Only wiping presses; the other tool motions cut, mix and pour in free
    space.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        :return: A plan moving the tool along the sampled waypoints while keeping it
            aligned with its target.
        """
        return sequential(
            [
                MoveTCPWaypointsAlignedMotion(
                    waypoints=self._waypoints,
                    arm=self.arm,
                    allow_gripper_collision=True,
                    alignment_pairs=self._alignment_pairs,
                    tip=self.tool.get_tool_frame(),
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                    desired_force=self.desired_force,
                    admittance_mass=self.admittance_mass,
                    admittance_damping=self.admittance_damping,
                )
            ]
        )


@dataclass(kw_only=True)
class MixingAction(ToolMotionAction):
    """
    Mix the contents of a container with a tool.
    """

    container: Body
    """
    The container (e.g., a bowl) whose contents are mixed.
    """

    mix_duration: float = 0.0
    """
    Total mixing time in seconds for a continuous stir loop.

    If not positive, a short spiral pattern is used instead.
    """

    def _build_tool_path(self) -> ToolPath:
        if self.mix_duration > 0.0:
            return build_container_path(
                self.container,
                pattern=MixingPattern.STIR,
                mix_duration=self.mix_duration,
            )
        return build_container_path(self.container, pattern=MixingPattern.SPIRAL)

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        return self.container.global_pose.to_homogeneous_matrix()

    @property
    def _alignment_target(self) -> Optional[Union[Body, Pose]]:
        return self.container


@dataclass(kw_only=True)
class CuttingAction(ToolMotionAction):
    """
    Cut a food object with a tool.
    """

    object_to_cut: Body
    """
    The object to cut.
    """

    technique: CuttingTechnique = CuttingTechnique.SAW
    """
    The cutting technique to use.
    """

    slice_thickness: Optional[float] = None
    """
    Target slice thickness in meters, controlling the spacing between cut anchors.

    Derived from ``number_of_cuts_on_local_x_axis`` and the object size if None.
    """

    number_of_cuts_on_local_x_axis: Optional[int] = None
    """
    Number of cut passes along the object's local X axis.

    Derived from ``slice_thickness`` and the object size if None.
    """

    slicing_priority: SlicingPriority = SlicingPriority.THICKNESS
    """
    Parameter that is kept when ``slice_thickness`` and
    ``number_of_cuts_on_local_x_axis`` do not both fit the object.
    """

    def _build_tool_path(self) -> ToolPath:
        return build_cutting_path(
            self.object_to_cut,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
            number_of_cuts_on_local_x_axis=self.number_of_cuts_on_local_x_axis,
            slicing_priority=self.slicing_priority,
        )

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        return self.object_to_cut.global_pose.to_homogeneous_matrix()

    @property
    def _alignment_target(self) -> Optional[Union[Body, Pose]]:
        return self.object_to_cut


@dataclass(kw_only=True)
class WipingAction(ToolMotionAction):
    """
    Wipe a surface or a patch around a target pose with a tool.
    """

    surface: Optional[Body] = None
    """
    The surface body to wipe.

    If None, ``target_pose`` is used instead.
    """

    target_pose: Optional[Pose] = None
    """
    Center pose of the wiping patch.

    Only used if ``surface`` is None.
    """

    technique: WipingTechnique = WipingTechnique.WIPE
    """
    The wiping technique to use.
    """

    length: float = 0.20
    """
    Length of the wiped patch along the sweep, in meters.
    """

    width: float = 0.0
    """
    Width of the wiped patch across the sweep, in meters.

    Zero wipes a single lane back and forth. A positive width lays parallel lanes side
    by side, a tool width apart, so a rectangle is covered rather than a line.
    """

    cycles: float = 1.0
    """
    Number of sweep cycles, for a patch of no width.
    """

    final_waypoint_success_tolerance: float = 0.08
    """
    Accept an unfinished motion as successful if the tool ends up within this distance
    in meters of the final waypoint.
    """

    def __post_init__(self):
        """
        :raises WipingTargetMissing: If neither a surface nor a target pose is given.
        """
        if self.surface is None and self.target_pose is None:
            raise WipingTargetMissing(self)

    def _build_tool_path(self) -> ToolPath:
        if self.surface is not None:
            if self.desired_force is None:
                return build_surface_path(self.surface, technique=self.technique)
            # Pressing puts the path on the surface and lets the admittance regulate how
            # hard; the default clearance would keep the tool just above it instead.
            return build_surface_path(
                self.surface, technique=self.technique, approach_clearance=0.0
            )
        if self.technique is WipingTechnique.SPREAD:
            if self.width > 0.0:
                return ToolPath(
                    [
                        ToolPathSegment(
                            kind=ToolPathSegmentKind.SWEEP,
                            duration=2.0,
                            local_curve=lambda tau: planar_raster_xy(
                                tau,
                                width=float(self.length),
                                height=float(self.width),
                                lanes=self._lanes(),
                            ),
                        )
                    ]
                )
            return ToolPath(
                [
                    ToolPathSegment(
                        kind=ToolPathSegmentKind.SWEEP,
                        duration=2.0,
                        local_curve=lambda tau: planar_sweep_x(
                            # Half, because a sweep is given its amplitude while
                            # :attr:`length` is the patch it has to cover.
                            tau,
                            length=0.5 * float(self.length),
                            cycles=max(1.0, float(self.cycles)),
                        ),
                    )
                ]
            )
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SPIRAL,
                    duration=2.0,
                    local_curve=lambda tau: planar_spiral_xy(
                        tau, r0=0.00, r1=0.12, cycles=2.5
                    ),
                )
            ]
        )

    def _lanes(self) -> int:
        """
        :return: How many lanes cover :attr:`width`, spaced so that the tool overlaps
            between them and leaves no strip untouched.
        """
        footprint = self.tool.root.collision.as_bounding_box_collection_in_frame(
            self.tool.root
        ).bounding_box()
        return max(2, math.ceil(self.width / footprint.width) + 1)

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        if self.surface is not None:
            return self.surface.global_pose.to_homogeneous_matrix()
        if self.target_pose.reference_frame is None:
            self.target_pose.reference_frame = self.world.root
        return self.target_pose.to_homogeneous_matrix()

    @property
    def _alignment_target(self) -> Optional[Union[Body, Pose]]:
        if self.surface is not None:
            return self.surface
        return self.target_pose

    def _perform_plan(self) -> None:
        """
        Perform the wiping plan, accepting an unfinished motion if the tool still
        reached the final waypoint.

        A wipe can also end early by succeeding: a press that stops getting anywhere ends
        the motion rather than running to the timeout. Reporting how far the tool got
        keeps that from looking like a wipe of the whole surface.
        """
        subplan = self.add_subplan(self.action_plan)
        try:
            subplan.perform()
        except MotionDidNotFinish:
            if not self._tool_reached_final_waypoint():
                raise
        self._report_how_far_the_tool_got()

    def _report_how_far_the_tool_got(self) -> None:
        """
        Log where the tool ended up relative to the path it was given.
        """
        remaining = self._distance_to_final_waypoint()
        if remaining <= float(self.final_waypoint_success_tolerance):
            logger.info(f"wiped to the end of the path, {remaining:.3f} m from it")
            return
        logger.warning(
            f"the wipe stopped {remaining:.3f} m short of the end of its path, which is "
            f"further than the {self.final_waypoint_success_tolerance:.3f} m that counts "
            f"as reaching it"
        )

    def _tool_reached_final_waypoint(self) -> bool:
        """
        :return: True if the tool's root ended up within the success tolerance of the
            final waypoint.
        """
        return self._distance_to_final_waypoint() <= float(
            self.final_waypoint_success_tolerance
        )

    def _distance_to_final_waypoint(self) -> float:
        """
        :return: How far the tool's root ended up from the last waypoint of its path.
        """
        tool_point = self.world.transform(
            self.tool.root.global_pose.to_position(), self.world.root
        )
        tool_xyz = np.asarray(tool_point.to_np(), dtype=float).reshape(-1)[:3]
        goal_point = self.world.transform(self._waypoints[-1], self.world.root)
        goal_xyz = np.asarray(goal_point.to_np(), dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(tool_xyz - goal_xyz))


@dataclass(kw_only=True)
class PouringAction(FullBodyControlledAction, HasTcpGoalThresholds):
    """
    Pour from a held source container into a target container by tilting the source next
    to the target's rim.
    """

    target_container: Body
    """
    The container that is poured into.
    """

    source_container: Tool
    """
    The held container that is poured from.
    """

    arm: Arms
    """
    The arm holding the source container.
    """

    tilt_angle: float = 1.85
    """
    Tilt angle in radians applied to the source container while pouring.
    """

    pour_side: Optional[Arms] = None
    """
    Robot-relative side of the target container to pour from.

    Defaults to the arm, so one-arm robots can still use either side's pouring geometry.
    """

    pour_side_offset: float = 0.0
    """
    Extra lateral offset in meters of the pour point from the target container's center.
    """

    pour_approach_offset: float = 0.0
    """
    Extra offset in meters away from the target container along the approach direction.
    """

    pour_height: float = 0.13
    """
    TCP height in meters above the target container for the pre-pour pose.
    """

    def _effective_pour_side(self) -> Arms:
        """
        :return: The requested pour side, or the pouring arm if none was requested.
        """
        if self.pour_side is None:
            return self.arm
        return self.pour_side

    def _mouth_height_above_tool_frame(self) -> float:
        """
        :return: Height in meters of the source container's opening above the arm's
            tool frame, measured along the tool frame's z axis.
        """
        tool_frame = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        tool_frame_T_source = self.world.compute_forward_kinematics_np(
            tool_frame, self.source_container.root
        )
        bounding_box = (
            self.source_container.root.visual.as_bounding_box_collection_in_frame(
                self.source_container.root
            ).bounding_box()
        )
        mouth_in_source = np.array(
            [
                0.5 * (bounding_box.min_x + bounding_box.max_x),
                0.5 * (bounding_box.min_y + bounding_box.max_y),
                bounding_box.max_z,
                1.0,
            ]
        )
        return float((tool_frame_T_source @ mouth_in_source)[2])

    def _approach_direction(
        self, target_pose: Pose, robot_pose: Pose
    ) -> Tuple[float, float]:
        """
        :return: The XY unit vector from the robot toward the target container,
            snapped to the target's nearest local axis so the pour never aims at a
            corner.
        """
        approach_x = float(target_pose.x) - float(robot_pose.x)
        approach_y = float(target_pose.y) - float(robot_pose.y)
        approach_norm = math.hypot(approach_x, approach_y)
        if approach_norm < 1e-6:
            approach_x, approach_y = 1.0, 0.0
        else:
            approach_x /= approach_norm
            approach_y /= approach_norm

        target_quaternion = [
            float(value) for value in target_pose.to_quaternion().to_np()
        ]
        target_rotation = Rotation.from_quat(target_quaternion)
        target_x_axis = target_rotation.apply([1, 0, 0])
        target_y_axis = target_rotation.apply([0, 1, 0])
        approach_vector = np.array([approach_x, approach_y, 0.0])
        candidates = [target_x_axis, -target_x_axis, target_y_axis, -target_y_axis]
        alignments = [np.dot(approach_vector, candidate) for candidate in candidates]
        best = candidates[int(np.argmax(alignments))]

        snapped_norm = math.hypot(float(best[0]), float(best[1]))
        if snapped_norm <= 1e-6:
            return approach_x, approach_y
        return float(best[0]) / snapped_norm, float(best[1]) / snapped_norm

    def _pour_poses(self) -> Tuple[Pose, Pose]:
        """
        :return: The pre-pour pose next to the target container's rim and the tilted
            pouring pose.
        """
        pour_side = self._effective_pour_side()
        target_pose = self.target_container.global_pose
        robot_pose = self.robot.root.global_pose

        approach_x, approach_y = self._approach_direction(target_pose, robot_pose)
        robot_right_x = approach_y
        robot_right_y = -approach_x
        side_sign = 1.0 if pour_side == Arms.RIGHT else -1.0

        side_offset = float(self.pour_side_offset) + math.sin(self.tilt_angle) * max(
            self._mouth_height_above_tool_frame(), 0.0
        )
        approach_offset = float(self.pour_approach_offset)

        pour_x = (
            float(target_pose.x)
            + side_sign * robot_right_x * side_offset
            - approach_x * approach_offset
        )
        pour_y = (
            float(target_pose.y)
            + side_sign * robot_right_y * side_offset
            - approach_y * approach_offset
        )
        pour_z = float(target_pose.z) + float(self.pour_height)

        yaw_to_target = math.atan2(
            float(target_pose.y) - pour_y, float(target_pose.x) - pour_x
        )
        base_rotation = Rotation.from_euler("z", yaw_to_target)
        if pour_side == Arms.LEFT:
            base_rotation = Rotation.from_euler("z", math.pi) * base_rotation

        signed_tilt_angle = (
            self.tilt_angle if pour_side == Arms.RIGHT else -self.tilt_angle
        )
        tilted_rotation = base_rotation * Rotation.from_euler("y", signed_tilt_angle)

        pre_pour_pose = self._pose_from_rotation(pour_x, pour_y, pour_z, base_rotation)
        pour_pose = self._pose_from_rotation(pour_x, pour_y, pour_z, tilted_rotation)
        return pre_pour_pose, pour_pose

    def _pose_from_rotation(
        self, x: float, y: float, z: float, rotation: Rotation
    ) -> Pose:
        """
        :param x: X position of the pose in the world frame.
        :param y: Y position of the pose in the world frame.
        :param z: Z position of the pose in the world frame.
        :param rotation: Orientation of the pose.
        :return: The pose in the world frame.
        """
        quat_x, quat_y, quat_z, quat_w = rotation.as_quat()
        return Pose.from_xyz_quaternion(
            pos_x=x,
            pos_y=y,
            pos_z=z,
            quat_x=quat_x,
            quat_y=quat_y,
            quat_z=quat_z,
            quat_w=quat_w,
            reference_frame=self.world.root,
        )

    @property
    def _action_plan(self) -> PlanNode:
        """
        :return: A plan moving the source container to the pre-pour pose and then
            tilting it into the pouring pose.
        """
        pre_pour_pose, pour_pose = self._pour_poses()
        return sequential(
            [
                MoveToolCenterPointMotion(
                    pre_pour_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    movement_type=MovementType.CARTESIAN,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveToolCenterPointMotion(
                    pour_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    movement_type=MovementType.CARTESIAN,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
            ]
        )
