"""
Driving the hand of the real HSRB through the grasp action of its gripper controller.

.. note:: Importing this module needs ``tmc_control_msgs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.plans.plan_transformation import ConfigurationTransformation
from coraplex.robot_plans.mixins import OperatesGripper
from giskardpy.motion_statechart.ros2_nodes.hsrb_hand import HSRBHandActionServerTask
from semantic_digital_twin.robots.hsrb import HSRBGripper

# %% driving the hand through its grasp action


@dataclass
class DriveTheHSRBHandThroughItsGraspAction(
    ConfigurationTransformation[OperatesGripper]
):
    """
    Brings the hand of an HSRB into its configurations through the grasp action of its
    gripper controller, instead of with goals for its finger joints, which no controller
    of the real HSRB is commanded with.
    """

    def is_applicable(self, plan_node: OperatesGripper) -> bool:
        return any(
            isinstance(end_effector, HSRBGripper)
            for end_effector in plan_node.robot.all_end_effectors
        )

    def apply(self, plan_node: OperatesGripper) -> None:
        plan_node.gripper_goal_type = HSRBHandActionServerTask


# %% running plans on the real HSRB


@dataclass(kw_only=True)
class RealHSRBPlanExecutor(RealRobotPlanExecutor):
    """
    Runs plans on the real HSRB, whose hand moves through the grasp action of its
    gripper controller: :attr:`plan_adaptations` is extended by
    :class:`DriveTheHSRBHandThroughItsGraspAction`.
    """

    def __post_init__(self) -> None:
        self.plan_adaptations = [
            DriveTheHSRBHandThroughItsGraspAction(),
            *self.plan_adaptations,
        ]
        super().__post_init__()
