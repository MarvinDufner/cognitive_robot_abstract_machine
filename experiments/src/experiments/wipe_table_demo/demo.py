"""
The HSRB wipes a kitchen table, regulating how hard it presses with its wrist
force/torque sensor.

The wipe itself is coraplex's :class:`~coraplex.robot_plans.actions.composite.tool_based.
WipingAction`; giving it a ``desired_force`` is what turns its geometric tool path into a
compliant one. The table is wider than the arm reaches, so the plan simply drives to each
side in turn and wipes again -- each wipe covers what it can and stops when it can get no
further.

``ExecutionType.REAL`` drives the real robot and takes the world from the running
controller, which must already be up (see
``giskardpy/middleware/ros2/scripts/iai_robots/hsr/iai_hsr_real_time.py``) together with
the wrench compensation node, so the sensor reports contact force rather than the
gripper's own weight. The default builds the kitchen locally and simulates the contact.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import ClassVar, List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType, WipingTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.view_manager import ViewManager
from giskardpy.middleware.ros2.input_synchronization import InputSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predetermined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DiningTable,
    Sponge,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

SPONGE_NAME = "wipe_sponge"
"""
Name of the sponge body bolted to the gripper.
"""

SPONGE_SCALE = Scale(0.08, 0.055, 0.03)
"""
Size of the sponge.
"""

APPROACH_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, y=5.705, z=0.0)
"""
Where the robot is put down in the kitchen, in front of the dining table.
"""

PRESS_FORCE = 15.0
"""
Contact force the admittance holds against the table, in N.
"""

STANDING_CLEARANCE = 0.05
"""
Gap left between the base and the table edge when standing at a side, in m.
"""

SIMULATED_CONTROL_PERIOD = 1 / 50
"""
Time between two simulated control cycles, in s, matching the rate coraplex simulates
at.
"""

CONTACT_STIFFNESS = 2000.0
"""
Stiffness of the simulated table surface, in N/m.
"""

MAXIMUM_CONTACT_FORCE = 50.0
"""
Largest force the simulated surface pushes back with, in N.

A control cycle is long enough that a fast approach registers as deep penetration, which
this model would otherwise report as a spike of many kilonewtons.
"""

CONTACT_VELOCITY_GAIN = 4000.0
"""
Extra contact stiffness per metre per second of approach speed, in N s/m^2.

Damps the simulated impact so the press settles instead of ringing.
"""

WAYPOINT_STRIDE = 10
"""
Keep every Nth point of the generated tool path.

The path is sampled at 0.4 mm, far finer than a wipe needs, and the trajectory advances
at most a couple of waypoints per control cycle, so the raw density alone decides how
long the motion takes. Thinning it costs no coverage.
"""

OUTWARD_NORMALS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
"""
The four sides of a table, as outward normals in its own frame.
"""

# %% simulated contact


@dataclass
class SimulatedContactWrench(InputSynchronizer):
    """
    Writes the wrench a table would exert on the tool into the force/torque annotation.

    Stands in for the sensor while no robot is attached, using a Hunt-Crossley contact
    model: penetration into the surface produces a force along the surface normal,
    stiffened by how fast the tool is still descending. On the robot the controller
    follows the sensor's own topic instead, and nothing here is used.
    """

    tool: Body = field(kw_only=True)
    """
    Body whose penetration into the surface is measured.
    """

    surface_height: float = field(kw_only=True)
    """
    Height of the wiped surface in the world frame, in m.
    """

    surface: Body = field(kw_only=True)
    """
    The wiped body, whose footprint bounds where contact is felt.
    """

    control_period: float = field(kw_only=True)
    """
    Time between two readings, in s, used to measure how fast the tool approaches.
    """

    stiffness: float = field(default=CONTACT_STIFFNESS, kw_only=True)
    """
    Stiffness of the simulated surface, in N/m.
    """

    velocity_gain: float = field(default=CONTACT_VELOCITY_GAIN, kw_only=True)
    """
    Extra stiffness per unit approach speed, in N s/m^2.
    """

    _penetration: float = field(init=False, default=0.0)
    """
    Penetration measured on the previous reading, in m.
    """

    def is_over_the_surface(self, tool_position: np.ndarray) -> bool:
        """
        :param tool_position: Where the tool is, in the world frame.
        :return: Whether the tool is above the surface rather than beside it, where
            there is nothing to touch.
        """
        box = self.surface.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()
        return (
            box.min_x <= tool_position[0] <= box.max_x
            and box.min_y <= tool_position[1] <= box.max_y
        )

    def apply(self) -> bool:
        """
        Write the contact wrench, and report that nothing has to be announced: a wrench
        leaves the kinematic state untouched.
        """
        sensor = ForceTorqueSensor.for_tip(self.world, self.tool)
        tool_position = self.world.compute_forward_kinematics_np(
            self.world.root, self.tool
        )[:3, 3]
        penetration = self.surface_height - float(tool_position[2])
        if not self.is_over_the_surface(tool_position):
            penetration = 0.0
        approach_speed = max(
            0.0, (penetration - self._penetration) / self.control_period
        )
        self._penetration = penetration

        force = np.zeros(3)
        if penetration > 0.0:
            magnitude = min(
                penetration * (self.stiffness + self.velocity_gain * approach_speed),
                MAXIMUM_CONTACT_FORCE,
            )
            world_R_sensor = self.world.compute_forward_kinematics_np(
                self.world.root, sensor.root
            )[:3, :3]
            force = world_R_sensor.T @ np.array([0.0, 0.0, magnitude])
        sensor.write_wrench(force, np.zeros(3))
        return False


# %% the demonstrations


@dataclass
class WipingDemonstration(RobotDemonstration, ABC):
    """
    A demonstration that wipes a surface with a sponge held in one hand.
    """

    arm: Arms = Arms.LEFT
    """
    Arm that holds the sponge.
    """

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SPONGE_NAME)

    def attach_sponge(self, world: World) -> Sponge:
        """
        Bolt the sponge to the gripper's tool frame and annotate it as a tool.

        :param world: World holding the robot.
        :return: The sponge, whose frame follows the wipe path.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        tool_frame = ViewManager.get_end_effector_view(self.arm, robot).tool_frame
        sponge_body = Body(
            name=PrefixedName(SPONGE_NAME),
            collision=ShapeCollection([Box(scale=SPONGE_SCALE)]),
            visual=ShapeCollection([Box(scale=SPONGE_SCALE)]),
        )
        sponge = Sponge(root=sponge_body)
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=tool_frame,
                    child=sponge_body,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=float(SPONGE_SCALE.z) / 2.0
                        )
                    ),
                )
            )
            world.add_semantic_annotations([sponge])
        return sponge

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the robot in ``world``.

        .. note:: A real run needs the ROS node in the context to reach the controller.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
            world_inputs=self.world_inputs(world),
        )

    @abstractmethod
    def world_inputs(self, world: World) -> List[InputSynchronizer]:
        """
        :param world: World the demonstration acts on.
        :return: Whatever feeds live readings into the world while the plan runs.
        """


@dataclass
class WipeTableDemonstration(WipingDemonstration):
    """
    The HSRB wipes the kitchen's dining table, one side at a time.
    """

    ros_node_name: ClassVar[str] = "wipe_table_demo_node"

    technique: WipingTechnique = WipingTechnique.SPREAD
    """
    How the tool covers the surface: lanes, a spiral or a shear.
    """

    sides: Optional[int] = None
    """
    How many of the usable sides to wipe from, or ``None`` for all of them.

    One placement reaches under half of this table, so covering it means driving round
    it. Set this to shorten a demonstration.
    """

    def build_simulated_world(self) -> World:
        """
        Put the robot on its drive in an otherwise empty world; the kitchen is added by
        :meth:`populate_scene`, which a real run skips.
        """
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=APPROACH_POSE,
                )
            ],
        ).to_domain_object()

    def populate_scene(self, world: World) -> None:
        """
        Add the kitchen and bolt the sponge into the gripper.
        """
        if not world.get_semantic_annotations_by_type(DiningTable):
            with world.modify_world():
                world.merge_world(KitchenEnvironment().get_world())
        self.attach_sponge(world)

    def build_plan(self, context: Context) -> PlanNode:
        """
        Drive to each side of the table in turn and wipe from there.
        """
        world = context.world
        # A software-driven base. Without this the collision rules discard every base
        # contact as an ignorable adjacency, so the base would not avoid the kitchen.
        for drive in world.get_connections_by_type(OmniDrive):
            drive.has_hardware_interface = True

        table = world.get_semantic_annotations_by_type(DiningTable)[0]
        sponge = world.get_semantic_annotations_by_type(Sponge)[0]

        steps = []
        for standing_pose in self.standing_poses(world, table)[: self.sides]:
            steps.append(NavigateAction(standing_pose))
            steps.append(
                WipingAction(
                    arm=self.arm,
                    tool=sponge,
                    surface=table.root,
                    technique=self.technique,
                    pointer_stride=WAYPOINT_STRIDE,
                    skip_unreachable=True,
                    standing_pose=standing_pose,
                    desired_force=Vector3(z=PRESS_FORCE),
                )
            )
        return sequential(steps, context=context)

    def world_inputs(self, world: World) -> List[InputSynchronizer]:
        """
        The readings written into the world before each simulated control cycle.

        :param world: World holding the table and the sponge.
        :return: A simulated contact model, or nothing on the real robot, where the
            controller follows the sensor's own topic.
        """
        if self.execution_type is ExecutionType.REAL:
            return []
        table = world.get_semantic_annotations_by_type(DiningTable)[0]
        return [
            SimulatedContactWrench(
                world=world,
                tool=world.get_body_by_name(SPONGE_NAME),
                surface=table.root,
                surface_height=self.surface_height(world, table),
                control_period=SIMULATED_CONTROL_PERIOD,
            )
        ]

    @staticmethod
    def surface_height(world: World, table: DiningTable) -> float:
        """
        :param world: World holding the table.
        :param table: The table whose top is wanted.
        :return: Height of the table top in the world frame, in m.
        """
        body = table.root
        top_in_table_frame = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .max_z
        )
        return (
            float(world.compute_forward_kinematics_np(world.root, body)[2, 3])
            + top_in_table_frame
        )

    def standing_poses(self, world: World, table: DiningTable) -> List[Pose]:
        """
        Where the base can stand to wipe, one pose per side of the table.

        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :return: Poses just clear of each edge, facing the table, nearest side first.
        """
        body = table.root
        extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        world_T_table = world.compute_forward_kinematics_np(world.root, body)
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        standoff = robot.mobile_base.footprint_radius + STANDING_CLEARANCE
        half_depth = 0.5 * extent.depth
        half_width = 0.5 * extent.width

        poses = []
        for normal_x, normal_y in OUTWARD_NORMALS:
            table_P_base = np.array(
                [
                    normal_x * (half_depth + standoff),
                    normal_y * (half_width + standoff),
                    0.0,
                    1.0,
                ]
            )
            world_P_base = world_T_table @ table_P_base
            table_yaw = math.atan2(world_T_table[1, 0], world_T_table[0, 0])
            across = extent.depth if normal_x else extent.width
            poses.append(
                (
                    Pose.from_xyz_rpy(
                        x=float(world_P_base[0]),
                        y=float(world_P_base[1]),
                        z=0.0,
                        yaw=table_yaw + math.atan2(-normal_y, -normal_x),
                        reference_frame=world.root,
                    ),
                    across,
                )
            )
        robot_position = world.compute_forward_kinematics_np(world.root, robot.root)[
            :2, 3
        ]

        def preference(entry) -> tuple:
            """
            Shallow sides first, since the arm has to reach across the table, and among
            equals the one the robot is already closest to.
            """
            pose, across = entry
            distance = float(np.linalg.norm(pose.to_np()[:2, 3] - robot_position))
            return across, distance

        usable = [
            entry
            for entry in poses
            if self.has_room_for_the_base(world, table, entry[0])
        ]
        return [pose for pose, _ in sorted(usable, key=preference)]

    def has_room_for_the_base(
        self, world: World, table: DiningTable, pose: Pose
    ) -> bool:
        """
        Whether the base fits where a side would be wiped from.

        The table itself is ignored: standing beside it is the point, and
        :data:`STANDING_CLEARANCE` already holds the base off its edge.

        :param world: World holding the table and whatever surrounds it.
        :param table: The table being wiped.
        :param pose: Where the base would stand.
        :return: Whether that spot is clear.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        base = robot.mobile_base
        base_box = base.bounding_box
        clearance = base.footprint_radius + STANDING_CLEARANCE
        standing = pose.to_np()[:2, 3]
        ignored = set(robot.bodies_with_collision) | set(
            world.get_kinematic_structure_entities_of_branch(table.root)
        )
        for body in world.bodies_with_collision:
            if body in ignored:
                continue
            box = body.collision.as_bounding_box_collection_in_frame(
                world.root
            ).bounding_box()
            # Only what stands where the base stands is in the way: the floor is below
            # it and the table top is above it.
            if box.max_z <= base_box.min_z or box.min_z >= base_box.max_z:
                continue
            gap_x = max(box.min_x - standing[0], 0.0, standing[0] - box.max_x)
            gap_y = max(box.min_y - standing[1], 0.0, standing[1] - box.max_y)
            if math.hypot(gap_x, gap_y) < clearance:
                return False
        return True


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    """
    WipeTableDemonstration(used_robot=HSRB, execution_type=execution_type).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.REAL)
