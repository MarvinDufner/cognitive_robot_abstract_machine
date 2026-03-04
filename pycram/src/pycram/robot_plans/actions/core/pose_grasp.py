from dataclasses import dataclass
from typing import Union, Iterable

from pycram.datastructures.enums import Arms
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.language import SequentialPlan
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.robot_plans.motions.pose_grasp import (
    PoseGraspMotion,
    RetractMotion,
    RetractDirection,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPose


@dataclass
class PoseGraspAction(ActionDescription):
    """
    Move the gripper to an object's annotated grasp pose and close the gripper.
    Requires the object to carry a HasGraspPose semantic annotation so that
    self.object.grasp_pose is guaranteed to exist.
    """

    object: HasGraspPose
    """The object to grasp."""

    arm: Arms
    """The arm to use for grasping."""

    collision_distance: float = 0.01
    """Minimum distance for collision avoidance."""

    pre_grasp_distance: float = 0.15
    """Distance to offset the pre-grasp pose along the negative z-axis of the grasp pose (in meters)."""

    def execute(self) -> None:
        SequentialPlan(
            self.context,
            MoveGripperMotion(gripper=self.arm, motion=GripperState.OPEN),
            PoseGraspMotion(
                arm=self.arm,
                grasp_pose=self.object.grasp_pose,
                object_bodies=list(self.object.bodies),
                collision_distance=self.collision_distance,
                pre_grasp_distance=self.pre_grasp_distance,
            ),
            MoveGripperMotion(gripper=self.arm, motion=GripperState.CLOSE),
        ).perform()

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        object: HasGraspPose,
        collision_distance: Union[Iterable[float], float] = 0.01,
        pre_grasp_distance: Union[Iterable[float], float] = 0.15,
    ) -> PartialDesignator["PoseGraspAction"]:
        return PartialDesignator[PoseGraspAction](
            PoseGraspAction,
            arm=arm,
            object=object,
            collision_distance=collision_distance,
            pre_grasp_distance=pre_grasp_distance,
        )


@dataclass
class PoseGraspAndLiftAction(ActionDescription):
    """
    Complete grasping sequence: open gripper, approach via grasp pose, close gripper,
    attach object, then retract.
    Requires the object to carry a HasGraspPose semantic annotation so that
    self.object.grasp_pose is guaranteed to exist.
    """

    object: HasGraspPose
    """The object to grasp and lift."""

    arm: Arms
    """The arm to use for grasping."""

    pre_grasp_distance: float = 0.15
    """Distance to offset the pre-grasp pose along the negative z-axis of the grasp pose (in meters)."""

    retract_distance: float = 0.1
    """Distance to retract after grasping (in meters)."""

    retract_direction: RetractDirection = RetractDirection.WORLD_Z
    """Direction to retract along after grasping."""

    max_retract_velocity: float = 0.2
    """Maximum velocity during retract."""

    def execute(self) -> None:
        hand = ViewManager.get_end_effector_view(self.arm, self.robot_view)

        SequentialPlan(
            self.context,
            PoseGraspActionDescription(
                object=self.object,
                arm=self.arm,
                pre_grasp_distance=self.pre_grasp_distance,
            ),
        ).perform()

        with self.world.modify_world():
            self.world.move_branch_with_fixed_connection(
                self.object.root, hand.tool_frame
            )

        SequentialPlan(
            self.context,
            RetractMotion(
                arm=self.arm,
                object_bodies=list(self.object.bodies),
                distance=self.retract_distance,
                direction=self.retract_direction,
                reference_velocity=self.max_retract_velocity,
            ),
        ).perform()

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        object: HasGraspPose,
        pre_grasp_distance: Union[Iterable[float], float] = 0.15,
        retract_distance: Union[Iterable[float], float] = 0.1,
        retract_direction: Union[
            Iterable[RetractDirection], RetractDirection
        ] = RetractDirection.WORLD_Z,
        max_retract_velocity: Union[Iterable[float], float] = 0.2,
    ) -> PartialDesignator["PoseGraspAndLiftAction"]:
        return PartialDesignator[PoseGraspAndLiftAction](
            PoseGraspAndLiftAction,
            arm=arm,
            object=object,
            pre_grasp_distance=pre_grasp_distance,
            retract_distance=retract_distance,
            retract_direction=retract_direction,
            max_retract_velocity=max_retract_velocity,
        )


PoseGraspActionDescription = PoseGraspAction.description
PoseGraspAndLiftActionDescription = PoseGraspAndLiftAction.description
