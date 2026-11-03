"""
The HSR opens the door of the washing machine in the kitchen environment, simulated in
MuJoCo.

The door starts cracked open. The HSR reaches into the crack at the grip recess under
the upper left of the door ring, one finger in front of the ring and one behind it, and
opens the door along its hinge. Holding on to the door while the hinge is driven to its
upper limit makes the gripper follow the hinge as it is modelled, so the run shows
whether the modelled hinge matches the door's mesh.

Run it as a script::

    python3 -m experiments.hsr_laundry_demo.demo
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import Pose, RotationMatrix, Vector3
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    WashingMachine,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class HSRLaundryDemonstration(RobotDemonstration):
    """
    The HSR opens the cracked open door of the washing machine by reaching behind it.
    """

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "hsr_laundry_demo_node"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

    door_crack_angle: float = 0.2
    """
    How far the door of the washing machine stands open when the run starts, in radians:

    wide enough for a finger to fit behind the door at its grip recess.
    """

    standing_distance: float = 0.55
    """
    How far the robot's base stands back from the grip recess, horizontally along the
    way the gripper reaches into it, in meters.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(WashingMachine))

    def populate_scene(self, world: World) -> None:
        """
        Crack the door of the washing machine open by :attr:`door_crack_angle`.
        """
        self._washing_machine_door(
            world
        ).root.parent_connection.position = self.door_crack_angle

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park the arm, drive up to the door and open it by its grip recess.
        """
        hsrb = context.require_extension(RobotAccess).robot
        [arm] = hsrb.all_arms
        grasp = self._grasp_behind(self._washing_machine_door(context.world))
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    NavigateAction(target_location=self._standing_pose_for(grasp)),
                    OpenAction(grasp=grasp, arm=arm),
                    ParkArmsAction(arms=hsrb.all_arms),
                ]
            )
        )
        return statechart

    @staticmethod
    def _washing_machine_door(world: World) -> Door:
        """
        :param world: The world holding the washing machine.
        :return: The door of the washing machine.
        """
        [washing_machine] = world.get_semantic_annotations_by_type(WashingMachine)
        [door] = washing_machine.doors
        return door

    @staticmethod
    def _grasp_behind(door: Door) -> GraspCandidate:
        """
        :param door: The door of the washing machine.
        :return: The grasp of the door's grip recess that reaches over the rim of the
            door ring toward its centre, its fingers closing across the ring, so one of
            them ends up behind the door.
        """
        handle = door.handle.root
        toward_the_ring_centre = Vector3(0, 0, -1, reference_frame=handle)
        across_the_ring = Vector3(1, 0, 0, reference_frame=handle)
        return GraspCandidate(
            door.handle,
            Pose(
                orientation=RotationMatrix.from_vectors(
                    x=toward_the_ring_centre, y=across_the_ring
                ).quaternion,
                reference_frame=handle,
            ),
        )

    def _standing_pose_for(self, grasp: GraspCandidate) -> Pose:
        """
        :param grasp: The grasp the robot takes.
        :return: Where the robot's base stands on the floor to take ``grasp``,
            :attr:`standing_distance` back from it along the horizontal part of the
            approach and facing along it.
        """
        world = grasp.graspable.root._world
        world_T_grasp = world.transform(grasp.grasp_pose, world.root).to_np()
        approach = world_T_grasp[:3, 0].copy()
        approach[2] = 0.0
        approach /= np.linalg.norm(approach)
        standing_position = world_T_grasp[:3, 3] - self.standing_distance * approach
        return Pose.from_xyz_rpy(
            x=standing_position[0],
            y=standing_position[1],
            yaw=np.arctan2(approach[1], approach[0]),
            reference_frame=world.root,
        )


# %% running it


def main() -> None:
    """
    Run the demonstration.
    """
    HSRLaundryDemonstration().run()


if __name__ == "__main__":
    main()
