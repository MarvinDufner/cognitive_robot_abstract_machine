"""
Tracy wipes the part of its own bench that its left arm reaches, with the sponge in that
hand, regulating how hard it presses with the left wrist's force/torque sensor.

The tool is brought over the patch, lowered until the sensor feels the bench, and only
then wiped, so where the surface really is is measured rather than assumed.

The execution type is the first command line argument, defaulting to ``REAL``::

    python3 demo.py PHYSICALLY_SIMULATED

``PHYSICALLY_SIMULATED`` performs the wipe against MuJoCo, which resolves the contact
itself, so how hard and how steadily the press holds can be measured without the robot.

.. todo:: The sensor is taken as the control program leaves it, zeroed at start. Once
    the tool changes orientation under force, run a wrench compensation node per arm and
    read its topic instead. See
    :class:`~semantic_digital_twin.robots.tracy.TracyWrenchTopic`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from typing_extensions import ClassVar, List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, WipingTechnique
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
)
from coraplex.robot_plans.motions.gripper import LowerUntilContactMotion
from experiments.wipe_table_demo.demo import (
    PRESS_FORCE,
    PRESS_TRAVEL,
    SPONGE_NAME,
    SimulatedContactWrench,
    WipingDemonstration,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

BENCH_NAME = "table"
"""The bench Tracy is mounted on, which is also the surface it wipes."""

TRACY_FLOOR_NAME = "floor"
"""Name of the body Tracy is bolted to in a simulated run."""

WIPED_PATCH_CENTRE = (0.72, -0.10)
"""Centre of the wiped patch, in the bench frame."""

WIPED_PATCH_LENGTH = 0.34
"""Length of the wiped patch along the bench's long axis, in m.

.. warning:: Clear of the camera pole's brace (out to x = 0.35) at one end and of the
    far arm at the other. Widening the patch means measuring both again.
"""

WIPED_PATCH_WIDTH = 0.44
"""Width of the wiped patch across the bench, in m: what the arm can hold the tool flat
on, not what the 1.6 m wide bench offers.
"""

APPROACH_HEIGHT = 0.08
"""How far above the bench the tool is brought before it is lowered onto it, in m."""

TURNING_HEIGHT = 0.30
"""How far above the bench the tool is turned to face it, in m.

The tool reaches a commanded position before it reaches a commanded orientation, so a
tool that turns on its way down meets the bench edge first. It turns up here instead and
holds that orientation for the rest of the plan.
"""

OBSTACLE_CLEARANCE = 0.05
"""Gap kept between the hand and anything standing on the bench, in m: the hand reaches
almost twice as far sideways as the sponge it holds.
"""

PRESS_DAMPING = Vector3(x=300.0, y=300.0, z=300.0)
"""Virtual damping the press yields with, per axis, in N s/m. The press chases the
surface at the force asked for divided by this, so 5 cm/s here.
"""

PRESS_MASS = Vector3(x=2.0, y=2.0, z=2.0)
"""Virtual mass the press yields with, per axis, in kg. Light, because the mass over the
damping is phase lag the loop does not need: 10 kg costs a third of the contact.
"""

PRESS_LEAD_TIME = 0.03
"""How far ahead of its own offset the press tracks, in s: about half of the 50 ms the
arm takes to reach a commanded height.
"""

WIPE_SPEED = 0.045
"""How fast the tool travels while wiping, in m/s.

The tendency to be thrown off a crest grows with the square of the speed, so this is the
speed at which the press keeps contact over an uneven surface as well as on a flat one.
"""

DESCENT_VELOCITY = 0.02
"""How fast the tool is lowered onto the bench, in m/s. Peak contact force grows with
the speed the tool arrives at, and the pad meets a crest on a line rather than on its
whole face, so arriving slowly is what keeps a touchdown off a crest from spiking.
"""

TOUCHDOWN_FORCE = 5.0
"""Force magnitude that ends the descent onto the bench, in N: above what the sensor
reports out of contact, far below what the arm can press with.
"""

DESCENT_OVERSHOOT = 0.02
"""How far below the bench the descent aims, in m. Contact ends the descent, so the goal
has to lie past the surface, and close enough to bound a descent that feels nothing.
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
        """
        Put Tracy on a floor it is bolted to; it brings its own bench.

        .. note:: The floor is not called ``map``, as a world root is by default, because
            Tracy's description has a link of that name and MuJoCo tells its bodies apart
            by name alone.
        """
        world = World.create_with_root_body(root_body_name=TRACY_FLOOR_NAME)
        RobotSpecification(semantic_annotation_type=self.used_robot).spawn(world)
        return world

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
        steps = [ParkArmsAction(arms=context.robot.all_arms)]
        steps.extend(self.build_approach(context, patch))
        steps.append(
            LowerUntilContactMotion(
                goal_point=Point3(
                    x=patch.to_np()[0, 3],
                    y=patch.to_np()[1, 3],
                    z=patch.to_np()[2, 3] - DESCENT_OVERSHOOT,
                    reference_frame=world.root,
                ),
                arm=self.wiping_arm(world),
                tip=sponge.get_tool_frame(),
                alignment_pairs=sponge.tool_alignment(patch),
                contact_force=TOUCHDOWN_FORCE,
                descent_velocity=DESCENT_VELOCITY,
                max_linear_velocity=DESCENT_VELOCITY,
                allow_gripper_collision=True,
            )
        )
        steps.append(
            WipingAction(
                arm=self.wiping_arm(world),
                tool=sponge,
                target_pose=patch,
                technique=WipingTechnique.WIPE,
                length=WIPED_PATCH_LENGTH,
                width=WIPED_PATCH_WIDTH,
                desired_force=Vector3(z=PRESS_FORCE),
                admittance_mass=PRESS_MASS,
                admittance_damping=PRESS_DAMPING,
                admittance_travel=PRESS_TRAVEL,
                admittance_lead_time=PRESS_LEAD_TIME,
                max_linear_velocity=WIPE_SPEED,
            )
        )
        return sequential(steps, context=context)

    def build_approach(
        self, context: Context, patch: Pose
    ) -> List[MoveManipulatorAction]:
        """
        :param context: Plan context holding the robot.
        :param patch: Centre of the patch, in the world frame.
        :return: Turning the tool onto the patch well above it, then coming straight
            down to the height the descent starts from.

        .. warning:: The approach has to point the sponge down. From the opposite
            orientation the wipe's alignment would have to flip the tool through the one
            pose where it cannot tell which way to turn.
        """
        end_effector = self.wiping_arm(context.world).end_effector
        return [
            MoveManipulatorAction(
                target_pose=self.pose_over_patch(context, patch, height),
                end_effector=end_effector,
                allow_gripper_collision=True,
            )
            for height in (TURNING_HEIGHT, APPROACH_HEIGHT)
        ]

    @staticmethod
    def pose_over_patch(context: Context, patch: Pose, height: float) -> Pose:
        """
        :param context: Plan context holding the world.
        :param patch: Centre of the patch, in the world frame.
        :param height: How far above the patch the tool stands, in m.
        :return: That pose, with the tool facing the patch.
        """
        return Pose.from_xyz_rpy(
            x=patch.to_np()[0, 3],
            y=patch.to_np()[1, 3],
            z=patch.to_np()[2, 3] + height,
            roll=math.pi,
            reference_frame=context.world.root,
        )

    def contact_model(self, world: World) -> SimulatedContactWrench:
        return SimulatedContactWrench(
            world=world,
            tool=world.get_body_by_name(SPONGE_NAME),
            surface=world.get_body_by_name(BENCH_NAME),
            surface_height=self.bench_height(world),
            control_period=SIMULATED_CONTROL_PERIOD,
        )

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


def main(execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED) -> None:
    """
    Run the demonstration.
    """
    TracyWipeTableDemonstration(used_robot=Tracy, execution_type=execution_type).run()


if __name__ == "__main__":
    main(
        execution_type=(
            ExecutionType[sys.argv[1]] if len(sys.argv) > 1 else ExecutionType.REAL
        )
    )
