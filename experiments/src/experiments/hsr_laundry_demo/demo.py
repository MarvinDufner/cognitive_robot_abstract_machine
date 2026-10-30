"""The HSR parks its arm in the kitchen environment next to a washing machine.

Run it as a script::

    python3 -m experiments.hsr_laundry_demo.demo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    WashingMachine,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

# %% the scene


class SceneFile(PathEnumeration):
    """The files the scene is built from, next to this demo."""

    WASHING_MACHINE = (
        Path(__file__).resolve().parent / "resources" / "washing_machine.stl"
    )
    """The mesh of the washing machine in meters, its origin at the center of
    its bottom and its front facing -X."""


class SceneBody(StrEnum):
    """The names of the scene's bodies."""

    WASHING_MACHINE = "washing_machine"


# %% demonstration


@dataclass
class HSRLaundryDemonstration(RobotDemonstration):
    """The HSR parks its arm in the kitchen environment in front of a washing
    machine."""

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "hsr_laundry_demo_node"

    world_T_washing_machine: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=2.8, y=-0.8
        )
    )
    """Where the washing machine stands in the kitchen, its front facing the
    robot."""

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5, y=-0.8
        )
    )
    """Where the robot starts, facing the washing machine."""

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
        """Add the kitchen environment and stand the washing machine in it."""
        kitchen = KitchenEnvironment().get_world()
        world.merge_world(
            kitchen, FixedConnection(parent=world.root, child=kitchen.root)
        )
        SemanticAnnotationWithRootSpecification(
            name=SceneBody.WASHING_MACHINE,
            semantic_annotation_type=WashingMachine,
            root_specification=BodySpecification.mesh(
                SceneBody.WASHING_MACHINE,
                str(SceneFile.WASHING_MACHINE),
                parent_T_self=self.world_T_washing_machine,
            ),
        ).spawn(world)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """Park the arm."""
        hsrb = context.require_extension(RobotAccess).robot
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=[ParkArmsAction(arms=hsrb.all_arms)]))
        return statechart


# %% running it


def main() -> None:
    """Run the demonstration."""
    HSRLaundryDemonstration().run()


if __name__ == "__main__":
    main()
