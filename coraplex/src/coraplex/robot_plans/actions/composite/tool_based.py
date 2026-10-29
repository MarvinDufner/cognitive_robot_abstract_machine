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
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    MovementType,
    PouringSide,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)
from coraplex.exceptions import (
    MissingWaypoints,
    WipingTargetMissing,
)
from coraplex.exceptions import ToolWithoutFootprint
from coraplex.plans.failures import MotionMadeNoProgress
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
    CartesianVelocityLimitParameters,
    HasTcpGoalThresholds,
    Press,
)
from coraplex.robot_plans.actions.composite.tool_paths import (
    ShearXYProfile,
    ToolPath,
    ToolPathSegment,
    build_container_path,
    build_cutting_path,
    build_surface_path,
    oscillatory_shear_xy_profiled,
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
class ToolMotionAction(
    FullBodyControlledAction,
    ABC,
    CartesianVelocityLimitParameters,
    HasTcpGoalThresholds,
):
    """
    An action that moves a tool along a sampled tool path while keeping the tool aligned
    with its target.
    """

    arm: Arm
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

    skip_unreachable: bool = False
    """
    Drop the waypoints the arm cannot reach from where the robot stands, so a path that
    starts out of reach still covers what it can.

    .. note:: The bound is purely kinematic, so a kept waypoint may still be blocked or
        need a posture the arm cannot hold.
    """

    press: Optional[Press] = None
    """
    How hard the tool is held against the target while it moves.

    ``None`` follows the tool path as generated, which keeps the tool just clear of the
    surface.
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
        path_frame = self._path_frame()
        _, points, _ = self._build_tool_path().sample(frame=path_frame)
        if self.press is not None:
            points = (
                points + path_frame.to_np()[:3, 2] * self.tool.contact_face_distance
            )
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

        .. note:: Where the robot stands now, which is where it will wipe from: an
            action is expanded when the steps before it have run, so a plan that drives
            somewhere first has already arrived.
        """
        standing = self.world.compute_forward_kinematics_np(
            self.world.root, self.robot.root
        )[:2, 3]
        distances = np.linalg.norm(points[:, :2] - standing, axis=1)
        return distances <= self.arm.maximum_reach()

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
                    press=self.press,
                    max_linear_velocity=self.max_linear_velocity,
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
        return self.container.global_pose.homogeneous_matrix

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
        return self.object_to_cut.global_pose.homogeneous_matrix

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
            # Pressing puts the path on the surface and lets the admittance regulate how
            # hard; the default clearance would keep the tool just above it instead.
            clearance = {} if self.press is None else {"approach_clearance": 0.0}
            return build_surface_path(
                self.surface, technique=self.technique, **clearance
            )
        if self.technique is WipingTechnique.SPREAD:
            return self._lane_path(self._tool_span)
        if self.technique is WipingTechnique.SHEAR:
            return self._circling_path(self._tool_span)
        return self._spiral_path(self._tool_span)

    @property
    def _tool_span(self) -> float:
        """
        :return: How wide a strip the tool covers, in m.
        :raises ToolWithoutFootprint: If the tool states no such strip.

        Only the tool's face is held against the surface; which way it points along the
        path is left to the arm, so a wider strip than its narrow side cannot be relied
        on.
        """
        shapes = self.tool.root.collision
        if not shapes:
            raise ToolWithoutFootprint(tool_name=str(self.tool.root.name))
        footprint = shapes.as_bounding_box_collection_in_frame(
            self.tool.root
        ).bounding_box()
        span = min(footprint.depth, footprint.width)
        if span <= 0.0:
            raise ToolWithoutFootprint(tool_name=str(self.tool.root.name))
        return span

    def _inscribed_radius(self, tool_span: float) -> float:
        """
        :param tool_span: How wide a strip the tool covers, in m.
        :return: The largest radius a path circling the patch's centre can have while
            the tool stays on the patch, in m.
        """
        across = self.width if self.width > 0.0 else self.length
        return max(0.0, 0.5 * min(self.length, across) - 0.5 * tool_span)

    def _lane_path(self, tool_span: float) -> ToolPath:
        """
        :param tool_span: How wide a strip the tool covers, in m.
        :return: Parallel lanes covering the patch, or a single lane swept back and
            forth where the patch has no width.

        The path is the tool's centre line, so it stops half a span short of the patch
        at every edge and the tool still covers it to the edges.
        """
        lane_length = max(0.0, self.length - tool_span)
        if self.width <= 0.0:
            return ToolPath(
                [
                    ToolPathSegment(
                        kind=ToolPathSegmentKind.SWEEP,
                        duration=2.0,
                        local_curve=lambda tau: planar_sweep_x(
                            # Half, because a sweep is given its amplitude while the
                            # lane is the line it has to cover.
                            tau,
                            length=0.5 * lane_length,
                            cycles=max(1.0, self.cycles),
                        ),
                    )
                ]
            )
        lane_span = max(0.0, self.width - tool_span)
        # One more lane than the spans between them, so that neighbouring lanes are at
        # most a tool apart and no strip is left untouched.
        lanes = max(2, math.ceil(lane_span / tool_span) + 1)
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SWEEP,
                    duration=2.0,
                    local_curve=lambda tau: planar_raster_xy(
                        tau, width=lane_length, height=lane_span, lanes=lanes
                    ),
                )
            ]
        )

    def _spiral_path(self, tool_span: float) -> ToolPath:
        """
        :param tool_span: How wide a strip the tool covers, in m.
        :return: An outward spiral over the largest disc the patch inscribes, turning
            often enough that its pitch stays within that span.

        .. note:: A disc is all a spiral reaches, so a rectangular patch keeps its
            corners.
        """
        radius = self._inscribed_radius(tool_span)
        turns = max(1.0, math.ceil(radius / tool_span))
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SPIRAL,
                    duration=2.0,
                    local_curve=lambda tau: planar_spiral_xy(
                        tau, r0=0.0, r1=radius, cycles=turns
                    ),
                )
            ]
        )

    def _circling_path(self, tool_span: float) -> ToolPath:
        """
        :param tool_span: How wide a strip the tool covers, in m.
        :return: Laps around the patch's centre, which is the shearing motion
            :func:`build_surface_path` gives a whole surface.
        """
        radius = self._inscribed_radius(tool_span)
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SHEAR,
                    duration=2.0,
                    local_curve=lambda tau: oscillatory_shear_xy_profiled(
                        tau,
                        ShearXYProfile(
                            shear_amp=radius,
                            shear_cycles=max(1.0, self.cycles),
                        ),
                    ),
                )
            ]
        )

    def _path_frame(self) -> HomogeneousTransformationMatrix:
        if self.surface is not None:
            return self.surface.global_pose.homogeneous_matrix
        if self.target_pose.reference_frame is None:
            self.target_pose.reference_frame = self.world.root
        return self.target_pose.homogeneous_matrix

    @property
    def _alignment_target(self) -> Optional[Union[Body, Pose]]:
        if self.surface is not None:
            return self.surface
        return self.target_pose

    def _perform_plan(self) -> None:
        """
        Perform the wiping plan, accepting a motion that gave up if the tool still
        reached the final waypoint, and reporting how far it got.
        """
        subplan = self.add_subplan(self.action_plan)
        try:
            subplan.perform()
        except MotionMadeNoProgress:
            if not self._tool_reached_final_waypoint():
                raise
        self._log_how_far_the_tool_got()

    def _log_how_far_the_tool_got(self) -> None:
        """
        Log where the tool ended up relative to the path it was given.
        """
        remaining = self._distance_to_final_waypoint()
        if remaining <= self.final_waypoint_success_tolerance:
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
        return (
            self._distance_to_final_waypoint() <= self.final_waypoint_success_tolerance
        )

    def _distance_to_final_waypoint(self) -> float:
        """
        :return: How far the tool's root ended up from the last waypoint of its path.
        """
        tool = (
            self.world.transform(self.tool.root.global_pose.position, self.world.root)
            .to_np()
            .flatten()[:3]
        )
        goal = (
            self.world.transform(self._waypoints[-1], self.world.root)
            .to_np()
            .flatten()[:3]
        )
        return float(np.linalg.norm(tool - goal))


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

    arm: Arm
    """
    The arm holding the source container.
    """

    tilt_angle: float = 1.85
    """
    Tilt angle in radians applied to the source container while pouring.
    """

    pour_side: Optional[PouringSide] = None
    """
    Robot-relative side of the target container to pour from.

    Defaults to the side of the pouring arm, so one-arm robots can still use either
    side's pouring geometry.
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

    def _effective_pour_side(self) -> PouringSide:
        """
        :return: The requested pour side, or the side of the pouring arm if none was
            requested.
        """
        if self.pour_side is not None:
            return self.pour_side
        if self.arm is self.robot.get_right_arm_if_specified():
            return PouringSide.RIGHT
        return PouringSide.LEFT

    def _mouth_height_above_tool_frame(self) -> float:
        """
        :return: Height in meters of the source container's opening above the arm's
            tool frame, measured along the tool frame's z axis.
        """
        tool_frame = self.arm.end_effector.tool_frame
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

        target_quaternion = [float(value) for value in target_pose.quaternion.to_np()]
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
        side_sign = 1.0 if pour_side == PouringSide.RIGHT else -1.0

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
        if pour_side == PouringSide.LEFT:
            base_rotation = Rotation.from_euler("z", math.pi) * base_rotation

        signed_tilt_angle = (
            self.tilt_angle if pour_side == PouringSide.RIGHT else -self.tilt_angle
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
