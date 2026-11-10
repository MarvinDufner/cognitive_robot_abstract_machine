from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    PouringSide,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)
from coraplex.exceptions import (
    MissingWaypoints,
    ToolWithoutFootprint,
    WipingTargetMissing,
)
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import (
    Action,
    ActionOfSteps,
    ParallelAction,
    SequentialAction,
    TryAllAction,
)
from coraplex.robot_plans.mixins import MovesToolCenterPoint, Press
from cramph.composites import Parallel, PausedUntilTrue
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.monitors.cartesian_monitors import PositionReached
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.monitors.force_monitors import ContactForceReached
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionTrajectory,
    HoldPose,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
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


@dataclass(kw_only=True, eq=False, repr=False)
class FullBodyControlledAction(Action, ABC):
    """
    An action that controls the robot's full body, so the base can support the arm
    motion.
    """

    @property
    def controlled_root(self) -> KinematicStructureEntity:
        """
        :return: The world root for a robot with a mobile base, since driving the base
            while manipulating moves the robot relative to the world; the robot's own
            root otherwise.
        """
        if isinstance(self.robot, HasMobileBase):
            return self.world.root
        return self.robot.root


@dataclass(kw_only=True, eq=False, repr=False)
class ToolMotionAction(
    FullBodyControlledAction, MovesToolCenterPoint, ActionOfSteps, ABC
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

    maximum_skip_ahead: int = 2
    """
    How many waypoints ahead the tool may already be heading for, so a path is followed
    as a continuous stroke rather than stopping at every point.
    """

    press: Press | None = None
    """
    How hard the tool is held against the target while it moves.

    ``None`` follows the tool path as generated, which keeps the tool just clear of the
    surface.
    """

    max_linear_velocity: float | None = None
    """
    Cap on the speed the tool travels at, in m/s.

    ``None`` leaves it to the arm.
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
    def _alignment_target(self) -> Body | Pose | None:
        """
        :return: The target the tool is aligned with during the motion.
        """

    @cached_property
    def _waypoints(self) -> list[Point3]:
        """
        :return: The sampled waypoints of the tool path in the world frame.
        """
        path_frame = self._path_frame()
        _, points, _ = self._build_tool_path().sample(frame=path_frame)
        if self.press is not None:
            points = (
                points + path_frame.to_np()[:3, 2] * self.tool.contact_face_distance
            )
        waypoints = [
            Point3(x=point[0], y=point[1], z=point[2], reference_frame=self.world.root)
            for point in points
        ][:: max(1, int(self.pointer_stride))]
        if not waypoints:
            raise MissingWaypoints(self)
        return waypoints

    @property
    def _alignment_pairs(self) -> list[AlignmentPair]:
        """
        :return: The normal pairs that keep the tool aligned with its target during
            the motion, or an empty list if there is no alignment target.
        """
        target = self._alignment_target
        if target is None:
            return []
        return self.tool.tool_alignment(target)

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the manipulator holding the tool to touch what the
            tool works on, for the whole stroke.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The nodes moving the tool along the sampled waypoints while keeping it
            aligned with its target, see :meth:`_tool_path_steps`.
        """
        return self._tool_path_steps()

    def _tool_path_steps(self) -> list[StatechartNode]:
        """
        :return: The nodes following the sampled waypoints with the tool together,
            holding every alignment the tool asks for while it moves, to be run side by
            side.
        """
        root = self.controlled_root
        tip = self.tool.get_tool_frame()
        alignments = [
            AlignPlanes(
                tip_link=tip,
                root_link=root,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value,
            )
            for pair in self._alignment_pairs
        ]
        return [
            self._follow_the_path(root, tip),
            *alignments,
            *self._velocity_limits(root, tip, self.max_linear_velocity, None),
            *self.keep_base_still(),
        ]

    def _follow_the_path(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> CartesianPositionTrajectory:
        """
        :param root: The frame the path is expressed in.
        :param tip: The tool frame following it.
        :return: The trajectory along the sampled waypoints, yielding to :attr:`press`
            where there is one and rigid otherwise.
        """
        arguments = dict(
            root_link=root,
            tip_link=tip,
            goal_points=self._waypoints,
            maximum_skip_ahead=self.maximum_skip_ahead,
            weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
            threshold=self.position_threshold,
        )
        if self.press is None:
            return CartesianPositionTrajectory(**arguments)
        return self.press.yielding_trajectory(**arguments)


@dataclass(kw_only=True, eq=False, repr=False)
class LowerUntilContactAction(
    FullBodyControlledAction, MovesToolCenterPoint, TryAllAction
):
    """
    Lower a tool until its force/torque sensor feels the surface.

    The goal sits below the surface, so the descent is ended by contact rather than by
    arriving: where the surface really is need not be known, only roughly where it is
    not.
    """

    goal_point: Point3
    """
    Point below the surface the tool is driven towards.
    """

    arm: Arm
    """
    The arm holding the tool.
    """

    tip: Body
    """
    The body that is lowered.
    """

    alignment_pairs: list[AlignmentPair] = field(default_factory=list)
    """
    Normal pairs kept aligned while lowering, so the tool keeps facing the surface it is
    lowered onto.
    """

    contact_force: float = 15.0
    """
    Force magnitude that counts as contact, in N.
    """

    descent_velocity: float = 0.02
    """
    How fast the tool is lowered, in m/s.

    Slow, because the tool travels on between the reading that shows contact and the
    cycle that acts on it.

    .. note:: This normalises the goal's error and does not bound the speed the tool
        arrives at. :attr:`max_linear_velocity` is what bounds it.
    """

    max_linear_velocity: float | None = None
    """
    Cap on the speed the tool descends at, in m/s.

    ``None`` leaves it to the arm.
    """

    allow_gripper_collision: bool = False
    """
    Whether the manipulator holding the tool may touch what it is lowered onto.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the manipulator holding the tool to touch the surface
            if :attr:`allow_gripper_collision`, none otherwise.
        """
        if not self.allow_gripper_collision:
            return []
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def keep_base_still(self) -> list[MotionStatechartNode]:
        """
        Holds the base wherever the robot has one, whatever its own setting says, see
        :meth:`~coraplex.robot_plans.actions.base.Action.keep_base_still`.

        The tool is lowered onto one point, and a base driving under it carries the tool
        off that point; the reach a driving base buys is of no use to a motion that only
        goes down.
        """
        if not isinstance(self.robot, HasMobileBase):
            return []
        return [
            HoldPose(
                name=f"{self.name}/hold base",
                root_link=self.world.root,
                tip_link=self.robot.root,
            )
        ]

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The descent, and the contact that ends it as the alternative to the
            tool arriving at a point it is not meant to reach.
        """
        root = self.controlled_root
        descent = Parallel(
            name=f"{self.name}/descent",
            nodes=[
                CartesianPosition(
                    root_link=root,
                    tip_link=self.tip,
                    goal_point=self.goal_point,
                    cruise_velocity=self.descent_velocity,
                    weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
                ),
                *[
                    AlignPlanes(
                        tip_link=self.tip,
                        root_link=root,
                        tip_normal=pair.tip_normal,
                        goal_normal=pair.goal_normal,
                        weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
                    )
                    for pair in self.alignment_pairs
                ],
                *self._velocity_limits(root, self.tip, self.max_linear_velocity, None),
                *self.keep_base_still(),
            ],
        )
        return [
            descent,
            ContactForceReached(
                name=f"{self.name}/contact",
                tip_link=self.tip,
                threshold=self.contact_force,
            ),
        ]


@dataclass(kw_only=True, eq=False, repr=False)
class MixingAction(ToolMotionAction, ParallelAction):
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
    def _alignment_target(self) -> Body | Pose | None:
        return self.container


@dataclass(kw_only=True, eq=False, repr=False)
class CuttingAction(ToolMotionAction, ParallelAction):
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

    slice_thickness: float | None = None
    """
    Target slice thickness in meters, controlling the spacing between cut anchors.

    Derived from ``number_of_cuts_on_local_x_axis`` and the object size if None.
    """

    number_of_cuts_on_local_x_axis: int | None = None
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
    def _alignment_target(self) -> Body | Pose | None:
        return self.object_to_cut


@dataclass(kw_only=True, eq=False, repr=False)
class WipingAction(ToolMotionAction, TryAllAction):
    """
    Wipe a surface or a patch around a target pose with a tool.
    """

    surface: Body | None = None
    """
    The surface body to wipe.

    If None, ``target_pose`` is used instead.
    """

    target_pose: Pose | None = None
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
    Accept a stalled motion as successful if the tool ended up within this distance in
    meters of the final waypoint.
    """

    stall_timeout: timedelta = timedelta(seconds=2)
    """
    How long the tool may make no progress along its path before the wipe is checked for
    having reached the final waypoint.

    .. note:: Has to be shorter than the stall timeout of the executor, which otherwise
        aborts the plan first.
    """

    def __post_init__(self):
        """
        :raises WipingTargetMissing: If neither a surface nor a target pose is given.
        """
        super().__post_init__()
        if self.surface is None and self.target_pose is None:
            raise WipingTargetMissing(self)

    def _build_tool_path(self) -> ToolPath:
        if self.surface is not None:
            # Pressing puts the path on the surface and lets the admittance regulate how
            # hard; the default clearance would keep the tool just above it instead.
            clearance = {} if self.press is None else {"approach_clearance": 0.0}
            return build_surface_path(
                self.surface,
                lane_spacing=self._tool_span,
                technique=self.technique,
                **clearance,
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
    def _alignment_target(self) -> Body | Pose | None:
        if self.surface is not None:
            return self.surface
        return self.target_pose

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The goal moving the tool along the sampled waypoints, and as an
            alternative to it, the tool having reached the final waypoint once that
            motion stalled, since the last stretch of a wipe often stalls against the
            surface.
        """
        tool_path = Parallel(nodes=self._tool_path_steps())
        return [
            tool_path,
            PausedUntilTrue(
                monitor=Stalled(monitored_node=tool_path, timeout=self.stall_timeout),
                monitored_node=PositionReached(
                    name=f"{self.name}/final waypoint reached",
                    root_link=self.world.root,
                    tip_link=self.tool.root,
                    goal_point=self._waypoints[-1],
                    threshold=self.final_waypoint_success_tolerance,
                ),
            ),
        ]


@dataclass(kw_only=True, eq=False, repr=False)
class PouringAction(FullBodyControlledAction, MovesToolCenterPoint, SequentialAction):
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

    pour_side: PouringSide | None = None
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
    ) -> tuple[float, float]:
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

    def _pour_poses(self) -> tuple[Pose, Pose]:
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

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the manipulator holding the source container to touch
            the rim it pours over.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        """
        :return: The goals moving the source container to the pre-pour pose and then
            tilting it into the pouring pose.
        """
        pre_pour_pose, pour_pose = self._pour_poses()
        return [
            self.tool_center_point_goal(
                pre_pour_pose,
                self.arm,
            ),
            self.tool_center_point_goal(
                pour_pose,
                self.arm,
            ),
        ]
