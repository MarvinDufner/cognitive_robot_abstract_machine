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

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import Pose, RotationMatrix, Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
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

    used_robot: type[AbstractRobot] = Tiago

    ros_node_name: str = "hsr_laundry_demo_node"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

    door_crack_angle: float = 0.2
    """
    How far the door of the washing machine stands open when the run starts, in radians:

    wide enough for a finger to fit behind the door at its grip recess.
    """

    standing_distance: float = 0.9
    """
    How far in front of the centre of the washing machine's footprint the robot's base
    stands, facing the machine, in meters.
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
        [door] = self._washing_machine(world).doors
        door.root.parent_connection.position = self.door_crack_angle

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park the arm, drive up to the door and open it by reaching over the rim of the
        door ring at its grip recess, the fingers closing across the ring so one of them
        ends up behind the door.
        """
        world = context.world
        hsrb = context.require_extension(RobotAccess).robot
        [arm] = hsrb.all_arms
        washing_machine = self._washing_machine(world)
        [door] = washing_machine.doors
        handle = door.handle.root
        grasp = GraspCandidate(
            door.handle,
            Pose(
                orientation=RotationMatrix.from_vectors(
                    x=Vector3(0, 0, -1, reference_frame=handle),
                    y=Vector3(1, 0, 0, reference_frame=handle),
                ).quaternion,
                reference_frame=handle,
            ),
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    NavigateAction(
                        target_location=Pose.from_xyz_rpy(
                            x=-self.standing_distance,
                            reference_frame=washing_machine.root,
                        )
                    ),
                    OpenAction(grasp=grasp, arm=arm),
                    ParkArmsAction(arms=hsrb.all_arms),
                ]
            )
        )
        return statechart

    @staticmethod
    def _washing_machine(world: World) -> WashingMachine:
        """
        :param world: The world holding the washing machine.
        :return: The washing machine.
        """
        [washing_machine] = world.get_semantic_annotations_by_type(WashingMachine)
        return washing_machine


# %% running it


def main() -> None:
    """
    Run the demonstration.
    """
    HSRLaundryDemonstration().run()


if __name__ == "__main__":
    main()
