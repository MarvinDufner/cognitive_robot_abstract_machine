"""
The HSR parks its arm in the kitchen environment next to a washing machine.

Run it as a script::

    python3 -m experiments.hsr_laundry_demo.demo
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    WashingMachine,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

# %% demonstration


@dataclass
class HSRLaundryDemonstration(RobotDemonstration):
    """
    The HSR parks its arm in the kitchen environment in front of a washing machine.
    """

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "hsr_laundry_demo_node"

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5, y=-0.8
        )
    )
    """
    Where the robot starts, facing the washing machine.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing, since :meth:`populate_scene` adds the kitchen
            environment.
        """
        return WorldSpecification(world_parser=None)

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=self.odom_T_robot_start,
        )

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(WashingMachine))

    def populate_scene(self, world: World) -> None:
        """
        Add the kitchen environment, with the washing machine's door and detergent
        drawer fully open.
        """
        kitchen = KitchenEnvironment().get_world()
        world.merge_world(
            kitchen, FixedConnection(parent=world.root, child=kitchen.root)
        )
        [washing_machine] = world.get_semantic_annotations_by_type(WashingMachine)
        for part in [*washing_machine.doors, *washing_machine.drawers]:
            joint = part.root.parent_connection
            joint.position = joint.dof.limits.upper.position

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park the arm.
        """
        hsrb = context.require_extension(RobotAccess).robot
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=[ParkArmsAction(arms=hsrb.all_arms)]))
        return statechart


# %% running it


def main() -> None:
    """
    Run the demonstration.
    """
    HSRLaundryDemonstration().run()


if __name__ == "__main__":
    main()
