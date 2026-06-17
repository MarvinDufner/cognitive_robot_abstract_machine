from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from pycram.robot_plans.motions.base import BaseMotion
from pycram.datastructures.enums import (
    Arms,
    MovementType,
    WaypointsMovementType,
)
from pycram.view_manager import ViewManager


def make_rule_for_allowing_collision_between_two_groups(
    bodies1: List[Body],
    bodies2: List[Body],
    robot: AbstractRobot,
    buffer_zone_distance: Optional[float] = None,
) -> UpdateTemporaryCollisionRules:
    """Create a temporary collision rule that allows collisions between two groups of bodies.

    If ``buffer_zone_distance`` is given it overrides the robot's default external-collision
    standoff for the duration of the motion; otherwise the robot's default is kept.
    """
    external_collisions = (
        AvoidExternalCollisions(robot=robot)
        if buffer_zone_distance is None
        else AvoidExternalCollisions(
            robot=robot, buffer_zone_distance=buffer_zone_distance
        )
    )
    return UpdateTemporaryCollisionRules(
        temporary_rules=[
            external_collisions,
            AllowCollisionBetweenGroups(
                body_group_a=[
                    b for b in bodies1 if b is not None and b.has_collision()
                ],
                body_group_b=[
                    b for b in bodies2 if b is not None and b.has_collision()
                ],
            ),
        ]
    )


def make_external_collision_buffer_rule(
    robot: AbstractRobot, buffer_zone_distance: float
) -> UpdateTemporaryCollisionRules:
    """Temporarily override the robot's external-collision buffer (soft standoff) distance."""
    return UpdateTemporaryCollisionRules(
        temporary_rules=[
            AvoidExternalCollisions(
                robot=robot, buffer_zone_distance=buffer_zone_distance
            )
        ]
    )


@dataclass(kw_only=True)
class ReachMotion(BaseMotion):
    """
    Moves the tool frame through a pose sequence (pre_grasp -> grasp), optionally avoiding
    collisions with everything except the bodies being manipulated.
    """

    arm: Arms
    """The arm performing the motion."""

    pose_sequence: List[Pose]
    """The [pre_grasp, grasp] poses the tool frame should move through."""

    allowed_collision_bodies: List[Body] = field(default_factory=list)
    """Bodies to allow collision with (typically the object being grasped)."""

    use_collision_avoidance: bool = False
    """Whether to avoid collisions (except with the allowed bodies) while reaching."""

    collision_buffer_distance: Optional[float] = None
    """Override for the external-collision buffer (soft standoff) distance in meters.
    None keeps the robot's default."""

    pre_grasp_threshold: float = 0.01
    """Goal threshold for the pre-grasp pose."""

    grasp_approach_velocity: float = 0.05
    """Linear velocity when moving from pre-grasp to grasp pose."""

    def perform(self):
        pass

    @property
    def _motion_chart(self) -> Task:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        tool_frame = hand.tool_frame
        pre_grasp_pose, grasp_pose = self.pose_sequence

        move_to_pre_grasp = CartesianPose(
            goal_pose=pre_grasp_pose,
            root_link=self.world.root,
            tip_link=tool_frame,
            threshold=self.pre_grasp_threshold,
        )
        move_to_grasp = CartesianPose(
            goal_pose=grasp_pose,
            root_link=self.world.root,
            tip_link=tool_frame,
            reference_linear_velocity=self.grasp_approach_velocity,
        )

        if not self.use_collision_avoidance:
            return Sequence([move_to_pre_grasp, move_to_grasp])

        pre_grasp_step = Parallel(
            [
                move_to_pre_grasp,
                ExternalCollisionAvoidance(robot=hand._robot),
                SelfCollisionAvoidance(robot=hand._robot),
            ],
            minimum_success=1,
        )
        if self.collision_buffer_distance is not None:
            pre_grasp_step = Parallel(
                [
                    make_external_collision_buffer_rule(
                        hand._robot, self.collision_buffer_distance
                    ),
                    pre_grasp_step,
                ]
            )
        grasp_step = self._grasp_step_with_collision_avoidance(move_to_grasp, hand)
        return Sequence([pre_grasp_step, grasp_step])

    def _grasp_step_with_collision_avoidance(self, task: Task, hand) -> Task:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulating_bodies = list({*arm.bodies, *hand.bodies})
        allow_rule = make_rule_for_allowing_collision_between_two_groups(
            manipulating_bodies,
            self.allowed_collision_bodies,
            robot=hand._robot,
            buffer_zone_distance=self.collision_buffer_distance,
        )
        motion = Parallel(
            [
                task,
                ExternalCollisionAvoidance(robot=hand._robot),
                SelfCollisionAvoidance(robot=hand._robot),
            ],
            minimum_success=1,
        )
        return Parallel([allow_rule, motion])


@dataclass
class MoveGripperMotion(BaseMotion):
    """
    Opens or closes the gripper
    """

    motion: GripperState
    """
    Motion that should be performed, either 'open' or 'close'
    """
    gripper: Arms
    """
    Name of the gripper that should be moved
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper is allowed to collide with something
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        arm = ViewManager().get_end_effector_view(self.gripper, self.robot)

        return JointPositionList(
            goal_state=arm.get_joint_state_by_type(self.motion),
            name=(
                "OpenGripper" if self.motion == GripperState.OPEN else "CloseGripper"
            ),
        )


@dataclass
class MoveToolCenterPointMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
    """
    movement_type: Optional[MovementType] = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        task = None
        if self.movement_type == MovementType.TRANSLATION:
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.to_position(),
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_CA,
            )
        else:
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_CA,
            )
        return task


@dataclass
class MoveTCPWaypointsMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    waypoints: List[Pose]
    """
    Waypoints the TCP should move along 
    """
    arm: Arms
    """
    Arm with the TCP that should be moved to the target
    """
    allow_gripper_collision: Optional[bool] = None
    """
    If the gripper can collide with something
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
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        nodes = [
            CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=pose,
                # threshold=0.005,
            )
            for pose in self.waypoints
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveManipulatorMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """

    manipulator: Manipulator
    """
    The Manipulator to move to the target pose
    """

    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something
    """

    @property
    def _motion_chart(self):
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        goal_pose = (
            self.target
            if self.robot.full_body_controlled
            else self.world.transform(self.target, root)
        )
        task = CartesianPose(
            root_link=root,
            tip_link=self.manipulator.tool_frame,
            goal_pose=goal_pose,
            threshold=0.005,
            name=self.__class__.__name__,
        )
        return task
