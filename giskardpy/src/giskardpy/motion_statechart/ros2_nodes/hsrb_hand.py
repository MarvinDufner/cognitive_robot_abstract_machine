"""
Driving the hand of the real HSRB through the ``tmc_control_msgs/GripperApplyEffort``
grasp action its gripper controller serves.

.. note:: Importing this module needs ``tmc_control_msgs``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from rclpy.task import Future
from tmc_control_msgs.action import GripperApplyEffort

from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from giskardpy.motion_statechart.exceptions import NoHSRBHandEffort
from giskardpy.motion_statechart.tasks.gripper import (
    MoveFingers,
    MovesGripperToConfiguration,
)
from giskardpy.motion_statechart.ros2_nodes.action_servers import ActionServerTask
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.hsrb import HSRBAction, HSRBGripper

logger = logging.getLogger(__name__)

# %% the torque of each hand state


class HSRBHandEffort(float, Enum):
    """
    The torque the hand's motor is driven with into each gripper state, in newton
    metres: positive opens the hand, negative closes it.
    """

    OPEN = 0.8
    """
    Opens the hand until it stalls at the end of its stroke.
    """

    CLOSE = -0.8
    """
    Closes the hand until it stalls on what it holds, or shut.
    """


# %% the grasp action server task


@dataclass(eq=False, repr=False, kw_only=True)
class HSRBHandActionServerTask(
    ActionServerTask[
        GripperApplyEffort,
        GripperApplyEffort.Goal,
        GripperApplyEffort.Result,
        GripperApplyEffort.Feedback,
    ],
    MovesGripperToConfiguration,
):
    """
    Drives the hand of the real HSRB with a torque through the grasp action of its
    gripper controller, until the hand stalls.
    """

    action_topic: str = field(default=HSRBAction.HAND_GRASP, kw_only=True)
    """
    The name the gripper controller serves the grasp action under.
    """

    message_type: type[GripperApplyEffort] = field(
        default=GripperApplyEffort, kw_only=True
    )
    """
    The ROS2 action interface type.
    """

    effort: float
    """
    The torque the hand's motor is driven with, in newton metres: positive opens the
    hand, negative closes it.
    """

    _goal_rejected: bool = field(init=False, default=False)
    """
    Whether the server rejected the goal last sent.
    """

    @classmethod
    def moving_gripper_to(
        cls,
        configuration: GripperConfiguration,
        stall_minimum_time: float | None = None,
        tolerate_stall: bool = False,
    ) -> StatechartNode:
        """
        Differs from :meth:`MovesGripperToConfiguration.moving_gripper_to` in driving
        the hand of an HSRB into the state of `configuration` with the
        :class:`HSRBHandEffort` of that state; any other gripper is driven as
        :meth:`MoveFingers.moving_gripper_to` drives it.

        :raises NoHSRBHandEffort: If no torque drives the hand into that state.
        """
        end_effector = configuration.end_effector
        if not isinstance(end_effector, HSRBGripper):
            return MoveFingers.moving_gripper_to(
                configuration, stall_minimum_time, tolerate_stall
            )
        match configuration.joint_state.state_type:
            case GripperState.OPEN:
                return cls(effort=HSRBHandEffort.OPEN)
            case GripperState.CLOSE:
                return cls(effort=HSRBHandEffort.CLOSE)
            case state_type:
                raise NoHSRBHandEffort(end_effector=end_effector, state_type=state_type)

    def build_action_goal(self) -> None:
        self._action_goal = GripperApplyEffort.Goal(effort=float(self.effort))
        self._goal_rejected = False

    def goal_response_callback(self, future: Future) -> None:
        """
        Differs from :meth:`ActionServerTask.goal_response_callback` in recording
        whether the server rejected the goal.
        """
        self._goal_rejected = not future.result().accepted
        super().goal_response_callback(future)

    def on_tick(self) -> ObservationStateValues:
        """
        :return: ``TRUE`` once the hand stalled, whether or not its torque settled at
            :attr:`effort`, ``FALSE`` once the server rejected the goal, and
            ``UNKNOWN`` until then.
        """
        if self._goal_rejected:
            return ObservationStateValues.FALSE
        if self._result is None:
            return ObservationStateValues.UNKNOWN
        return ObservationStateValues.TRUE
