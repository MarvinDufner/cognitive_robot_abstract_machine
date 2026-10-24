from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, Task
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.admittance_tasks import (
    AdmittanceCartesianTrajectory,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.monitors.force_monitors import ContactForceReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.robot_part_mixins import (
    HasMobileBase,
    HasTwoFingers,
)
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from coraplex.exceptions import (
    FingerWidthNeedsPhysics,
    MissingWaypoints,
)
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans.mixins import (
    CartesianVelocityLimitParameters,
    GripperStallToleranceParameters,
    HasTcpGoalThresholds,
)
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.datastructures.enums import (
    MovementType,
    WaypointsMovementType,
)


@dataclass
class MoveGripperMotion(BaseMotion, GripperStallToleranceParameters):
    """
    Opens or closes the gripper.
    """

    motion: GripperState
    """
    Motion that should be performed, either 'open' or 'close'.
    """

    gripper: EndEffector
    """
    The gripper that should be moved.
    """

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        name = "OpenGripper" if self.motion == GripperState.OPEN else "CloseGripper"
        goal_state = self.gripper.get_joint_state_by_type(self.motion)
        joint_task = JointPositionList(goal_state=goal_state, name=name)

        done_node = joint_task
        if self.tolerate_stall:
            stall_monitor = LocalMinimumReached(
                degrees_of_freedom=[
                    connection.raw_dof for connection in goal_state.connections
                ],
                minimum_time=(
                    self.stall_minimum_time
                    if self.stall_minimum_time is not None
                    else 1.0
                ),
                measure_from_own_start=True,
            )
            done_node = Parallel(
                [joint_task, stall_monitor], minimum_success=1, name=name
            )

        accompanying_nodes: List[MotionStatechartNode] = []
        if self.finger_velocity is not None:
            accompanying_nodes.append(
                JointVelocityLimit(
                    connections=list(goal_state.connections),
                    max_velocity=self.finger_velocity,
                )
            )
        if self.allow_gripper_collision:
            accompanying_nodes.extend(
                self._only_allow_gripper_collision_rules(self.gripper)
            )
        if not accompanying_nodes:
            return done_node
        return Parallel([done_node, *accompanying_nodes], name=name)


@dataclass
class MoveFingersToWidthMotion(BaseMotion):
    """
    Closes a two-finger gripper until its fingertips stand a given distance apart,
    rather than driving it to a gripper state of its own.

    Closing a gripper all the way onto an object that is not centred between the
    fingers wedges it sideways instead of holding it, so an object held by contact alone
    is better met at its own width. Giskard is given the distance between the fingertips
    as the goal and finds the joint positions that realise it, which is what keeps this
    independent of how a gripper's fingers are driven.

    ..note:: This only means anything where something pushes back. Where the world is
        moved kinematically a held object hangs off the tool frame however the fingers
        stand, and :class:`MoveGripperMotion` says what is meant more directly.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers close.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces are to stand, in metres.
    """

    squeeze: float = 0.0
    """
    How far past that width each fingertip is sent, in metres, so the fingers press into
    an object rather than merely touching it.

    Keep it small, since it is commanded penetration into a body that does not yield.
    """

    threshold: float = 0.0005
    """
    How close, in metres, the fingertips have to come to the goal for the motion to
    count as done: well under a squeeze, so the squeeze is not lost to the tolerance.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self) -> Task:
        execution_type = GiskardExecutable.execution_type
        if execution_type is None or not execution_type.objects_push_back:
            raise FingerWidthNeedsPhysics(execution_type)
        fingertips = self.gripper.fingertips
        return CartesianPosition(
            root_link=fingertips.finger,
            tip_link=fingertips.thumb,
            goal_point=fingertips.finger_P_thumb(
                max(0.0, self.width - 2 * self.squeeze)
            ),
            name="MoveFingersToWidth",
            threshold=self.threshold,
        )


@dataclass
class LowerUntilContactMotion(BaseMotion, CartesianVelocityLimitParameters):
    """
    Lower a tool until its force/torque sensor feels the surface.

    The goal sits below the surface, so the descent is ended by contact rather than by
    arriving: where the surface really is need not be known, only roughly where it is not.
    """

    goal_point: Point3
    """
    Point below the surface the tool is driven towards.
    """

    arm: Arm
    """
    Arm holding the tool.
    """

    tip: Optional[Body] = None
    """
    Body that is lowered.

    Defaults to the arm's tool frame.
    """

    alignment_pairs: List[AlignmentPair] = field(default_factory=list)
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

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip_link = self._resolve_tip(self.arm, self.tip)
        root_link = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        tasks = [
            CartesianPosition(
                root_link=root_link,
                tip_link=tip_link,
                goal_point=self.goal_point,
                reference_velocity=self.descent_velocity,
                weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
                name="LowerUntilContact",
            )
        ]
        tasks.extend(
            AlignPlanes(
                tip_link=tip_link,
                root_link=root_link,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
            )
            for pair in self.alignment_pairs
        )
        if self.max_linear_velocity is not None:
            tasks.append(
                CartesianPositionVelocityLimit(
                    root_link=root_link,
                    tip_link=tip_link,
                    max_linear_velocity=self.max_linear_velocity,
                )
            )
        motion_statechart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm.end_effector)
            if self.allow_gripper_collision
            else []
        )
        motion_statechart_nodes.extend(self.keep_base_still())
        motion_statechart_nodes.append(
            Parallel(
                [
                    Parallel(tasks),
                    ContactForceReached(
                        tip_link=tip_link, threshold=self.contact_force
                    ),
                ],
                minimum_success=1,
            )
        )
        return Parallel(motion_statechart_nodes)


@dataclass
class MoveToolCenterPointMotion(
    BaseMotion, CartesianVelocityLimitParameters, HasTcpGoalThresholds
):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    target: Pose
    """
    Target pose to which the TCP should be moved.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved to the target.
    """

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something.
    """

    movement_type: Optional[MovementType] = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    def _velocity_limit_nodes(self, root: Body, tip: Body) -> List[Task]:
        """
        :return: The :class:`CartesianPositionVelocityLimit`/
            :class:`CartesianRotationVelocityLimit` nodes requested via
            :attr:`max_linear_velocity`/:attr:`max_angular_velocity`, if any.
        """
        nodes = []
        if self.max_linear_velocity is not None:
            nodes.append(
                CartesianPositionVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_linear_velocity=self.max_linear_velocity,
                )
            )
        if (
            self.max_angular_velocity is not None
            and self.movement_type != MovementType.TRANSLATION
        ):
            nodes.append(
                CartesianRotationVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_angular_velocity=self.max_angular_velocity,
                )
            )
        return nodes

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        if self.movement_type == MovementType.TRANSLATION:
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.position,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                threshold=self.resolved_position_threshold(),
            )
        else:
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                translation_threshold=self.resolved_position_threshold(),
                orientation_threshold=self.resolved_orientation_threshold(),
            )
        accompanying_nodes: List[MotionStatechartNode] = list(
            self._velocity_limit_nodes(root, tip)
        )
        accompanying_nodes.extend(self.keep_base_still())
        if self.allow_gripper_collision:
            accompanying_nodes.extend(
                self._only_allow_gripper_collision_rules(self.arm.end_effector)
            )
        if not accompanying_nodes:
            return task
        return Parallel([task, *accompanying_nodes], name="MoveTCP")


@dataclass
class MoveTCPWaypointsMotion(BaseMotion, HasTcpGoalThresholds):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    waypoints: List[Pose]
    """
    Waypoints the TCP should move along.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved to the target.
    """

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something.
    """

    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        task_kwargs = dict(root_link=root, tip_link=tip)
        if self.position_threshold is not None:
            task_kwargs["translation_threshold"] = self.position_threshold
        if self.orientation_threshold is not None:
            task_kwargs["orientation_threshold"] = self.orientation_threshold
        nodes = [
            CartesianPose(
                goal_pose=pose,
                **task_kwargs,
            )
            for pose in self.waypoints
        ]
        waypoints = Sequence(nodes=nodes)
        hold_base = self.keep_base_still()
        if not hold_base:
            return waypoints
        return Parallel([waypoints, *hold_base])


@dataclass
class MoveTCPWaypointsAlignedMotion(
    BaseMotion, CartesianVelocityLimitParameters, HasTcpGoalThresholds
):
    """
    Moves the tool center point (TCP) of the robot along waypoints while keeping the
    given plane alignments.
    """

    waypoints: List[Point3]
    """
    Waypoints the TCP should move along.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved along the waypoints.
    """

    alignment_pairs: List[AlignmentPair] = field(default_factory=list)
    """
    Normal pairs kept aligned during the motion.
    """

    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something.
    """

    tip: Optional[Body] = None
    """
    The body that should follow the waypoints.

    Defaults to the arm's tool frame.
    """

    admittance_mass: Optional[Vector3] = None
    """
    Virtual mass the press yields with, per axis, in kg.

    ``None`` keeps the task's own default.
    """

    admittance_damping: Optional[Vector3] = None
    """
    Virtual damping the press yields with, per axis, in N s/m.

    ``None`` keeps the task's own default.
    """

    desired_force: Optional[Vector3] = None
    """
    Contact force to hold against the surface while following the waypoints, in the
    waypoints' own frame. ``None`` follows them rigidly.

    .. note:: Pressing needs a force/torque sensor on the chain to the tip.
    """

    admittance_lead_time: float = 0.0
    """
    How far ahead of the press's own offset the tracked point is placed, in s, to
    compensate the delay before the arm reaches a commanded pose.
    """

    admittance_travel: Optional[Vector3] = None
    """
    How far the press may yield from the waypoints, per axis, in m.

    ``None`` keeps the task's own default.

    .. note:: Zero on an axis keeps the tip rigid along it, which an axis that only
        feels friction needs: yielding to friction drifts the tip until it stalls.
    """

    look_ahead_distance: Optional[float] = None
    """
    How far along the waypoints ahead of the tip the tracked point is placed, in m,
    which is what sets how fast the tool travels.

    ``None`` keeps the task's own default.
    """

    def perform(self):
        return

    def _upright_torso_task(self, tip_link: Body, root_link: Body) -> AlignPlanes:
        """
        :return: A task that keeps Justin's torso upright during the motion.
        """
        torso_tip = self.robot.mobile_base.torso.tip
        return AlignPlanes(
            tip_link=torso_tip,
            root_link=root_link,
            tip_normal=Vector3.X(torso_tip),
            goal_normal=Vector3.Z(root_link),
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
        )

    @property
    def _motion_chart(self):
        if not self.waypoints:
            raise MissingWaypoints(self)

        tip_link = self._resolve_tip(self.arm, self.tip)
        root_link = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        trajectory_kwargs = dict(
            root_link=root_link,
            tip_link=tip_link,
            goal_points=self.waypoints,
            maximum_skip_ahead=2,
            weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
            name="MoveTCPWaypointsAligned",
        )
        if self.position_threshold is not None:
            trajectory_kwargs["threshold"] = self.position_threshold
        if self.look_ahead_distance is not None:
            trajectory_kwargs["look_ahead_distance"] = self.look_ahead_distance
        if self.desired_force is None:
            tasks = [CartesianPositionTrajectory(**trajectory_kwargs)]
        else:
            if self.admittance_mass is not None:
                trajectory_kwargs["mass"] = self.admittance_mass
            if self.admittance_damping is not None:
                trajectory_kwargs["damping"] = self.admittance_damping
            if self.admittance_travel is not None:
                trajectory_kwargs["maximum_offset"] = self.admittance_travel
            trajectory_kwargs["lead_time"] = self.admittance_lead_time
            tasks = [
                AdmittanceCartesianTrajectory(
                    desired_force=self.desired_force, **trajectory_kwargs
                )
            ]
        tasks.extend(
            AlignPlanes(
                tip_link=tip_link,
                root_link=root_link,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value,
            )
            for pair in self.alignment_pairs
        )
        if isinstance(self.robot, Justin):
            tasks.append(self._upright_torso_task(tip_link, root_link))
        if self.max_linear_velocity is not None:
            tasks.append(
                CartesianPositionVelocityLimit(
                    root_link=root_link,
                    tip_link=tip_link,
                    max_linear_velocity=self.max_linear_velocity,
                )
            )
        motion_statechart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm.end_effector)
            if self.allow_gripper_collision
            else []
        )
        motion_statechart_nodes.extend(self.keep_base_still())
        # No local-minimum monitor: a single cycle of near-zero velocity is the first
        # bounce off a stiff surface. The executable's no-progress monitor ends a press
        # that is really stuck.
        motion_statechart_nodes.append(Parallel(tasks))
        return Parallel(motion_statechart_nodes)


@dataclass
class MoveManipulatorMotion(BaseMotion, HasTcpGoalThresholds):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    target: Pose
    """
    Target pose to which the TCP should be moved.
    """

    end_effector: EndEffector
    """
    The end effector to move to the target pose.
    """

    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something.
    """

    @property
    def _motion_chart(self):
        robot = self.robot
        full_body_controlled = (
            robot.mobile_base.full_body_controlled
            if isinstance(robot, HasMobileBase)
            else False
        )
        root = self.world.root if full_body_controlled else robot.root
        task = CartesianPose(
            root_link=root,
            tip_link=self.end_effector.tool_frame,
            goal_pose=self.target,
            translation_threshold=self.resolved_position_threshold(),
            orientation_threshold=self.resolved_orientation_threshold(),
            binding_policy=GoalBindingPolicy.Bind_on_start,
            name=self.__class__.__name__,
        )
        hold_base = self.keep_base_still()
        if not hold_base:
            return task
        return Parallel([task, *hold_base])
