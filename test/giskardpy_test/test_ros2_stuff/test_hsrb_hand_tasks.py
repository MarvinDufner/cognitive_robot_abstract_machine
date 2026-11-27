"""
The task driving the hand of the real HSRB through its grasp action, skipped where
``tmc_control_msgs`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pytest

pytest.importorskip("tmc_control_msgs")

from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from tmc_control_msgs.action import GripperApplyEffort

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import RealTimePacer, StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import NoHSRBHandEffort
from giskardpy.motion_statechart.tasks.gripper import MoveFingers
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.hsrb_hand import (
    HSRBHandActionServerTask,
    HSRBHandEffort,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperStateConfiguration,
)
from semantic_digital_twin.robots.hsrb import HSRB, HSRBGripper, HSRBJoint
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World

# %% gripper controller stand-in


class HandActionName(StrEnum):
    """
    The action name the gripper controller stand-in serves under.
    """

    GRASP = "hsrb_hand_test/grasp"


@dataclass
class RecordingGraspServer:
    """
    Answers every accepted grasp goal as the gripper controller does once the hand
    stalled, and records the goals.
    """

    accepts_goals: bool = True
    """
    Whether the server accepts the goals it receives.
    """

    succeeds: bool = True
    """
    Whether the hand's torque settles at the commanded one, which the controller reports
    as success, and otherwise as an abort.
    """

    received_efforts: list[float] = field(default_factory=list)
    """
    The efforts of the goals the server executed, in order.
    """

    def accept_or_reject(self, goal: GripperApplyEffort.Goal) -> GoalResponse:
        """
        :return: Whether the server takes on `goal`, as :attr:`accepts_goals` says.
        """
        return GoalResponse.ACCEPT if self.accepts_goals else GoalResponse.REJECT

    def execute(self, goal: ServerGoalHandle) -> GripperApplyEffort.Result:
        """
        Records the goal's effort and ends it stalled.
        """
        self.received_efforts.append(goal.request.effort)
        if self.succeeds:
            goal.succeed()
        else:
            goal.abort()
        return GripperApplyEffort.Result(effort=goal.request.effort, stalled=True)

    def serve(self, node: Node) -> ActionServer:
        """
        :return: The action server answering grasp goals through :meth:`execute`.
        """
        return ActionServer(
            node,
            GripperApplyEffort,
            HandActionName.GRASP,
            self.execute,
            goal_callback=self.accept_or_reject,
        )


def run_until_observed(
    task: HSRBHandActionServerTask,
    observation: ObservationStateValues,
    rclpy_node: Node,
) -> None:
    """
    Ticks a statechart holding `task` until `task` observes `observation`.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(node=task)
    if observation == ObservationStateValues.TRUE:
        statechart.add_node(node=EndMotion.when_true(node=task))
    else:
        statechart.add_node(node=EndMotion.when_false(node=task))
    executor = StatechartExecutor(
        context=statechart.context,
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl()],
        pacer=RealTimePacer(),
    )
    executor.compile(statechart=statechart)
    executor.tick_until_end()


# %% helpers


def hand_of(world: World) -> HSRBGripper:
    """
    :return: The hand of the HSRB in `world`.
    """
    return world.get_semantic_annotations_by_type(HSRB)[0].all_arms[0].end_effector


# %% reaching a hand state


@pytest.mark.parametrize(
    "state_type, effort",
    [
        (GripperState.OPEN, HSRBHandEffort.OPEN),
        (GripperState.CLOSE, HSRBHandEffort.CLOSE),
    ],
    ids=["open", "close"],
)
def test_the_hand_is_driven_into_a_state_with_the_effort_of_that_state(
    state_type: GripperState, effort: HSRBHandEffort, hsr_world_copy: World
) -> None:
    configuration = hand_of(hsr_world_copy).default_configuration(state_type)

    task = HSRBHandActionServerTask.moving_gripper_to(configuration)

    assert type(task) is HSRBHandActionServerTask
    assert task.effort == effort


def test_a_hand_state_without_an_effort_raises(hsr_world_copy: World) -> None:
    hand = hand_of(hsr_world_copy)
    motor = hsr_world_copy.get_connection_by_name(HSRBJoint.HAND_MOTOR)
    configuration = GripperStateConfiguration(
        end_effector=hand,
        joint_state=JointState.from_mapping(
            {motor: 0.5}, state_type=GripperState.MEDIUM
        ),
    )

    with pytest.raises(NoHSRBHandEffort):
        HSRBHandActionServerTask.moving_gripper_to(configuration)


def test_a_gripper_of_another_robot_is_driven_with_a_finger_goal(
    pr2_world_copy: World,
) -> None:
    configuration = pr2_world_copy.get_semantic_annotations_by_type(PR2)[
        0
    ].left_arm.end_effector.default_configuration(GripperState.CLOSE)

    goal = HSRBHandActionServerTask.moving_gripper_to(configuration)

    assert type(goal) is type(MoveFingers.moving_gripper_to(configuration))


# %% talking to the grasp action server


@pytest.mark.parametrize("succeeds", [True, False], ids=["settled", "unsettled"])
def test_the_task_is_done_once_the_hand_stalled(
    succeeds: bool, rclpy_node: Node
) -> None:
    gripper_controller = RecordingGraspServer(succeeds=succeeds)
    server = gripper_controller.serve(rclpy_node)
    task = HSRBHandActionServerTask(
        action_topic=HandActionName.GRASP, effort=HSRBHandEffort.CLOSE
    )

    run_until_observed(task, ObservationStateValues.TRUE, rclpy_node)
    server.destroy()

    assert gripper_controller.received_efforts == [task.effort]


def test_the_task_fails_once_the_server_rejects_the_goal(rclpy_node: Node) -> None:
    server = RecordingGraspServer(accepts_goals=False).serve(rclpy_node)
    task = HSRBHandActionServerTask(
        action_topic=HandActionName.GRASP, effort=HSRBHandEffort.CLOSE
    )

    run_until_observed(task, ObservationStateValues.FALSE, rclpy_node)
    server.destroy()

    assert task.on_tick() == ObservationStateValues.FALSE
