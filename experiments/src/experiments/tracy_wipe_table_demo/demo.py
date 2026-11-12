"""
Tracy wipes the part of its own bench that its left arm reaches, regulating how hard it
presses with the left wrist's force/torque sensor.

The tool is lowered until the sensor feels the bench and only then wiped, so where the
surface really is is measured rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from coraplex.datastructures.enums import WipingTechnique
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
)
from cramph.composites import Sequence
from giskardpy.motion_statechart.tasks.admittance_tasks import Press
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.wipe_table_demo.demo import WipingDemonstration
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.tracy import Tracy, TracyFrame
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from krrood_spatial_math.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TracyWipeTableDemonstration(WipingDemonstration):
    """
    Tracy wipes its bench with its left hand.
    """

    ros_node_name: str = "tracy_wipe_table_demo_node"
    """
    Name of the node this demonstration talks to ROS under.
    """

    floor_name: str = "floor"
    """
    Name of the body Tracy is bolted to in a simulated run.

    .. note:: Not ``map``, as a world root is by default, because Tracy's description
        has a link of that name and MuJoCo tells its bodies apart by name alone.
    """

    bench_name: str = TracyFrame.BENCH
    """
    Name of the bench Tracy is mounted on, which is also the surface it wipes.
    """

    patch_centre: Point3 = field(default_factory=lambda: Point3(x=0.72, y=-0.10))
    """
    Centre of the wiped patch, in the bench frame, in m.
    """

    patch_length: float = 0.34
    """
    Length of the wiped patch along the bench's long axis, in m.

    .. warning:: Clear of the camera pole's brace (out to x = 0.35) at one end and of
        the far arm at the other. Widening the patch means measuring both again.
    """

    patch_width: float = 0.44
    """
    Width of the wiped patch across the bench, in m: what the arm can hold the tool flat
    on, not what the 1.6 m wide bench offers.
    """

    obstacle_clearance: float = 0.05
    """
    Gap kept between the hand and anything standing on the bench, in m: the hand reaches
    almost twice as far sideways as the sponge it holds.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        A bare floor for Tracy to be bolted to; it brings its own bench.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(semantic_annotation_type=self.used_robot)

    def build_simulated_world(self) -> World:
        """
        Spawns Tracy before the scene is populated, because the sponge is bolted into
        its gripper, see
        :meth:`~coraplex.demonstrations.RobotDemonstration.build_simulated_world`.
        """
        world = World.create_with_root_body(root_body_name=self.floor_name)
        self.robot_specification().spawn(world)
        self.populate_scene(world)
        return world

    def populate_scene(self, world: World) -> None:
        """
        Bolt the sponge into the gripper.

        Tracy brings its own bench.
        """
        self.attach_sponge(world)

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Reach over the patch, feel for the bench, then wipe the patch.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        sponge = world.get_semantic_annotations_by_type(Sponge)[0]
        patch = self.touchdown_pose(world)
        steps = [ParkArmsAction(arms=robot.all_arms)]
        steps.extend(self.build_approach(context, patch))
        steps.append(self.build_touchdown(context, patch))
        steps.append(
            WipingAction(
                arm=self.wiping_arm(world),
                tool=sponge,
                target_pose=patch,
                technique=self.technique,
                length=self.patch_length,
                width=self.patch_width,
                press=self.press,
                max_linear_velocity=self.wipe_speed,
            )
        )
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

    def wiped_surface(self, world: World) -> Body:
        return world.get_body_by_name(self.bench_name)

    def surface_height(self, world: World) -> float:
        """
        Takes the tallest of the shapes that span the bench rather than the tallest
        shape, see
        :meth:`~experiments.wipe_table_demo.demo.WipingDemonstration.surface_height`,
        because the bench carries a fixture as well as its top.
        """
        bench = self.wiped_surface(world)
        boxes = list(bench.collision.as_bounding_box_collection_in_frame(world.root))
        top = max(boxes, key=lambda box: box.max_x - box.min_x)
        return float(top.max_z)

    def touchdown_pose(self, world: World) -> Pose:
        """
        :param world: World holding the bench.
        :return: The centre of the wiped patch, on the bench top, in the world frame.
        """
        bench = world.get_body_by_name(self.bench_name)
        centre = world.transform(
            Point3(
                x=self.patch_centre.x,
                y=self.patch_centre.y,
                reference_frame=bench,
            ),
            world.root,
        ).to_np()
        return Pose.from_xyz_rpy(
            x=float(centre[0]),
            y=float(centre[1]),
            z=self.surface_height(world),
            reference_frame=world.root,
        )


def main(
    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor,
    technique: WipingTechnique = TracyWipeTableDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan.
    :param technique: How the tool covers the patch.
    """
    TracyWipeTableDemonstration(
        used_robot=Tracy, executor_type=executor_type, technique=technique
    ).run()


if __name__ == "__main__":
    main()
