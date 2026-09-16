"""
Tracy wipes the part of its own bench that its left arm reaches, with the sponge in that
hand, regulating how hard it presses with the left wrist's force/torque sensor.

The tool is brought over the patch, lowered until the sensor feels the bench, and only
then wiped, so where the surface really is is measured rather than assumed.

.. todo:: The sensor is taken as the control program leaves it, zeroed at start. Once
    the tool has to change orientation under force, the load has to be removed properly:
    run a wrench compensation node per arm and read its topic instead. See
    :class:`~semantic_digital_twin.robots.tracy.TracyWrenchTopic`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import ClassVar, List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, WipingTechnique, Arms
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
)
from coraplex.robot_plans.motions.gripper import LowerUntilContactMotion
from coraplex.view_manager import ViewManager
from experiments.wipe_table_demo.demo import (
    PRESS_FORCE,
    SPONGE_NAME,
    SimulatedContactWrench,
    WipingDemonstration,
)
from giskardpy.middleware.ros2.input_synchronization import InputSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

BENCH_NAME = "table"
"""The bench Tracy is mounted on, which is also the surface it wipes."""

WIPED_PATCH_CENTRE = (0.69, -0.10)
"""Centre of the wiped patch, in the bench frame."""

WIPED_PATCH_LENGTH = 0.40
"""Length of the wiped patch along the bench's long axis, in m.

Starts past the brace the camera pole stands on, which lies flat on the bench top and
reaches out to x = 0.35, keeping :data:`OBSTACLE_CLEARANCE` from it, and stops short of
the far arm. Both ends are measured from the bodies themselves, so widening the patch
means checking them again.
"""

WIPED_PATCH_WIDTH = 0.44
"""Width of the wiped patch across the bench, in m.

The bench is 1.6 m wide, but the arm can only hold the tool flat on it from roughly
0.35 m on its own side of the bench to 0.15 m across, so this covers what the arm
reaches rather than what the bench offers.
"""

APPROACH_HEIGHT = 0.08
"""How far above the bench the tool is brought before it is lowered onto it, in m.

Far enough to be clear of the surface wherever it really is, close enough that the slow
descent that follows does not take long.
"""

UNMODELLED_ADAPTER_LENGTH = 0.06
"""Length of the adapter between the flange and the gripper, in m.

.. todo:: Tracy's description is missing this piece, so every frame of the gripper sits
    that much too close to the flange. Remove this once the description has it. Guessing
    high is the safe direction: the descent feels for the surface either way, whereas
    guessing low presses the tool through it by the difference.
"""

OBSTACLE_CLEARANCE = 0.05
"""Gap kept between the hand and anything standing on the bench, in m.

The hand reaches almost twice as far sideways as the sponge it holds, it is free to spin
about the surface normal while it wipes, and it is driven along the patch rather than
tracked onto it exactly, so a patch the sponge alone clears is not enough.
"""

DESCENT_OVERSHOOT = 0.02
"""How far below the bench the descent aims, in m.

Contact is what ends the descent, so the goal has to lie past the surface; keeping it
close bounds how far the tool would travel if nothing were ever felt.
"""

SIMULATED_CONTROL_PERIOD = 1 / 50
"""Time between two simulated control cycles, in s."""


@dataclass
class TracyWipeTableDemonstration(WipingDemonstration):
    """
    Tracy wipes its bench with its left hand.
    """

    ros_node_name: ClassVar[str] = "tracy_wipe_table_demo_node"

    @property
    def unmodelled_tool_length(self) -> float:
        return UNMODELLED_ADAPTER_LENGTH

    def build_simulated_world(self) -> World:
        return WorldSpecification(
            world_parser=None,
            robots=[RobotSpecification(semantic_annotation_type=self.used_robot)],
        ).to_domain_object()

    def populate_scene(self, world: World) -> None:
        """
        Bolt the sponge into the gripper. Tracy brings its own bench.
        """
        self.attach_sponge(world)

    def build_plan(self, context: Context) -> PlanNode:
        """
        Reach over the patch, feel for the bench, then wipe the patch.
        """
        world = context.world
        sponge = world.get_semantic_annotations_by_type(Sponge)[0]
        patch = self.wiped_pose(world)
        steps = [ParkArmsAction(arm=Arms.BOTH), self.build_approach(context, patch)]
        steps.append(
            LowerUntilContactMotion(
                goal_point=Point3(
                    x=patch.to_np()[0, 3],
                    y=patch.to_np()[1, 3],
                    z=patch.to_np()[2, 3] - DESCENT_OVERSHOOT,
                    reference_frame=world.root,
                ),
                arm=self.arm,
                tip=sponge.get_tool_frame(),
                alignment_pairs=sponge.tool_alignment(patch),
                allow_gripper_collision=True,
            )
        )
        steps.append(
            WipingAction(
                arm=self.arm,
                tool=sponge,
                target_pose=patch,
                technique=WipingTechnique.SPREAD,
                length=WIPED_PATCH_LENGTH,
                width=WIPED_PATCH_WIDTH,
                desired_force=Vector3(z=PRESS_FORCE),
            )
        )
        return sequential(steps, context=context)

    def build_approach(self, context: Context, patch: Pose) -> PlanNode:
        """
        :param context: Plan context holding the robot.
        :param patch: Centre of the patch, in the world frame.
        :return: Bringing the tool over the patch, pointing at it. The tool goes there
            first because the descent and the wipe both start where they press, and
            because the wipe aligns the sponge's own axis with the surface: starting from
            the opposite orientation would ask the arm to flip through the one pose where
            that alignment cannot tell which way to turn.
        """
        return MoveManipulatorAction(
            target_pose=Pose.from_xyz_rpy(
                x=patch.to_np()[0, 3],
                y=patch.to_np()[1, 3],
                z=patch.to_np()[2, 3] + APPROACH_HEIGHT,
                roll=math.pi,
                reference_frame=context.world.root,
            ),
            end_effector=ViewManager.get_end_effector_view(self.arm, context.robot),
            allow_gripper_collision=True,
        )

    def world_inputs(self, world: World) -> List[InputSynchronizer]:
        """
        :param world: World holding the bench and the sponge.
        :return: A simulated contact model, or nothing on the real robot.
        """
        if self.execution_type is ExecutionType.REAL:
            return []
        return [
            SimulatedContactWrench(
                world=world,
                tool=world.get_body_by_name(SPONGE_NAME),
                surface=world.get_body_by_name(BENCH_NAME),
                surface_height=self.bench_height(world),
                control_period=SIMULATED_CONTROL_PERIOD,
            )
        ]

    @staticmethod
    def bench_height(world: World) -> float:
        """
        :param world: World holding the bench.
        :return: Height of the bench top in the world frame, in m.
        """
        bench = world.get_body_by_name(BENCH_NAME)
        # The bench carries a fixture as well as its top, so the top is the tallest of
        # the shapes that span it rather than the tallest shape.
        boxes = list(bench.collision.as_bounding_box_collection_in_frame(world.root))
        top = max(boxes, key=lambda box: box.max_x - box.min_x)
        return float(top.max_z)

    def wiped_pose(self, world: World) -> Pose:
        """
        :param world: World holding the bench.
        :return: The centre of the wiped patch, on the bench top, in the world frame.
        """
        world_T_bench = world.compute_forward_kinematics_np(
            world.root, world.get_body_by_name(BENCH_NAME)
        )
        world_P_centre = world_T_bench @ np.array(
            [WIPED_PATCH_CENTRE[0], WIPED_PATCH_CENTRE[1], 0.0, 1.0]
        )
        return Pose.from_xyz_rpy(
            x=world_P_centre[0],
            y=world_P_centre[1],
            z=self.bench_height(world),
            reference_frame=world.root,
        )


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """
    Run the demonstration.
    """
    TracyWipeTableDemonstration(used_robot=Tracy, execution_type=execution_type).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.REAL)
