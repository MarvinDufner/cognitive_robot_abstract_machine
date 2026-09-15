"""
Tracy wipes the half of its own bench that its left arm faces with the sponge in that
hand, regulating how hard it presses with the left wrist's force/torque sensor.

.. todo:: Nothing publishes Tracy's wrench topics yet, so a real run needs a wrench
    compensation node per arm before the press is meaningful. See
    :class:`~semantic_digital_twin.robots.tracy.TracyWrenchTopic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import ClassVar, List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from coraplex.view_manager import ViewManager
from experiments.wipe_table_demo.demo import (
    PRESS_FORCE,
    SPONGE_NAME,
    SimulatedContactWrench,
    WAYPOINT_STRIDE,
    WipingDemonstration,
)
from giskardpy.middleware.ros2.input_synchronization import InputSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

BENCH_NAME = "table"
"""The bench Tracy is mounted on, which is also the surface it wipes."""

WIPED_SPOTS = ((0.28, 0.0), (0.52, 0.0))
"""Centres of the patches the left hand wipes, in the bench frame.

The left arm is mounted off the low end of the bench's long axis, so these lie in the
half it faces, clear of the fixture standing on it. One wipe covers a patch of a fixed
size, so they are laid end to end; they follow that axis rather than spreading sideways
because that is where the arm can hold the tool flat on the bench.
"""

APPROACH_HEIGHT = 0.15
"""How far above the bench the hand is brought before the wipe starts, in m.

The arm parks well away from the spot, and one motion only gets so many control cycles,
so the reaching is done first and the wipe starts where it presses.
"""

SIMULATED_CONTROL_PERIOD = 1 / 50
"""Time between two simulated control cycles, in s."""


@dataclass
class TracyWipeTableDemonstration(WipingDemonstration):
    """
    Tracy wipes its bench with its left hand.
    """

    ros_node_name: ClassVar[str] = "tracy_wipe_table_demo_node"

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
        Wipe the bench patch by patch.

        .. todo:: Each patch is wiped around a pose, because a wipe of a *chosen part* of
            a surface has nowhere to say which part: ``WipingAction`` either takes a whole
            body or a pose, and the patch size is fixed inside ``build_surface_path``.
            Covering a named region in one wipe needs that region on the action.
        """
        plan = []
        for spot in self.wiped_poses(context.world):
            plan.extend(self.build_patch_plan(context, spot))
        return sequential(plan, context=context)

    def build_patch_plan(self, context: Context, spot: Pose) -> List[PlanNode]:
        """
        :param context: Plan context holding the world and the robot.
        :param spot: Centre of the patch, in the world frame.
        :return: Bringing the hand over the patch, then wiping it. The hand is brought
            there first because the wipe follows its waypoints in order and would
            otherwise spend its cycles travelling to the first one.
        """
        end_effector = ViewManager.get_end_effector_view(self.arm, context.robot)
        above = Pose.from_xyz_rpy(
            x=spot.to_np()[0, 3],
            y=spot.to_np()[1, 3],
            z=spot.to_np()[2, 3] + APPROACH_HEIGHT,
            reference_frame=context.world.root,
        )
        return [
            MoveManipulatorAction(
                target_pose=above,
                end_effector=end_effector,
                allow_gripper_collision=True,
            ),
            WipingAction(
                arm=self.arm,
                tool=context.world.get_semantic_annotations_by_type(Sponge)[0],
                target_pose=spot,
                pointer_stride=WAYPOINT_STRIDE,
                desired_force=Vector3(z=PRESS_FORCE),
            ),
        ]

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

    def wiped_poses(self, world: World) -> List[Pose]:
        """
        :param world: World holding the bench.
        :return: The centre of every patch that is wiped, in the world frame.
        """
        world_T_bench = world.compute_forward_kinematics_np(
            world.root, world.get_body_by_name(BENCH_NAME)
        )
        height = self.bench_height(world)
        poses = []
        for spot in WIPED_SPOTS:
            world_P_spot = world_T_bench @ np.array([spot[0], spot[1], 0.0, 1.0])
            poses.append(
                Pose.from_xyz_rpy(
                    x=world_P_spot[0],
                    y=world_P_spot[1],
                    z=height,
                    reference_frame=world.root,
                )
            )
        return poses


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """
    Run the demonstration.
    """
    TracyWipeTableDemonstration(used_robot=Tracy, execution_type=execution_type).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.REAL)
