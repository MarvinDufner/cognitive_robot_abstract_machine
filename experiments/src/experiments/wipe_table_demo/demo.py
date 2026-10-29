"""
The HSRB wipes a kitchen table, regulating how hard it presses with its wrist
force/torque sensor.

The wipe itself is coraplex's :class:`~coraplex.robot_plans.actions.composite.tool_based.
WipingAction`; giving it a ``press`` is what turns its geometric tool path into a
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
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta

import numpy as np
from typing_extensions import ClassVar, List, Optional, Tuple

from krrood.ormatic.utils import classproperty

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, WipingTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.mixins import Press
from semantic_digital_twin.adapters.sensors.force_torque import (
    PhysicallySimulatedContactWrench,
    ToolContactWrench,
)
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Arm, ForceTorqueSensor
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
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class TableSide(Enum):
    """
    The four sides of a table, as outward normals in its own frame.
    """

    FRONT = (1.0, 0.0)
    BACK = (-1.0, 0.0)
    LEFT = (0.0, 1.0)
    RIGHT = (0.0, -1.0)


@dataclass(frozen=True)
class TableSideStanding:
    """
    Where the base would stand to wipe from one side of a table.
    """

    pose: Pose
    """
    Where the base stands, facing the table.
    """

    across: float
    """
    How far the arm has to reach across the table from there, in m.
    """

    def preference(self, robot_position: np.ndarray) -> Tuple[float, float]:
        """
        :param robot_position: Where the robot stands now, in the world frame.
        :return: What orders this side against the others: shallow sides first, since
            the arm has to reach across the table, and among equals the one the robot
            is already closest to.
        """
        return self.across, float(
            np.linalg.norm(self.pose.to_np()[:2, 3] - robot_position)
        )


# %% simulated contact


@dataclass
class SimulatedContactWrench(ToolContactWrench):
    """
    What a flat surface of known height presses the tool with, modelled rather than
    simulated: penetration into the surface produces a force along the surface normal,
    stiffened by how fast the tool is still descending (Hunt-Crossley).

    Cheap enough to run a kinematically simulated plan against, where nothing resolves
    contacts.
    """

    surface_height: Optional[float] = field(default=None, kw_only=True)
    """
    Height of the wiped surface in the world frame, in m.

    ``None`` takes the top of :attr:`surface`, which is right for a surface that is one
    slab and wrong for one carrying a fixture taller than it.
    """

    surface: Body = field(kw_only=True)
    """
    The wiped body, whose footprint bounds where contact is felt.
    """

    control_period: float = field(kw_only=True)
    """
    Time between two readings, in s, used to measure how fast the tool approaches.
    """

    stiffness: float = field(default=2000.0, kw_only=True)
    """
    Stiffness of the simulated surface, in N/m.

    .. warning:: A control cycle is long enough that a fast approach registers as deep
        penetration, which this model would otherwise report as a spike of many
        kilonewtons; :attr:`maximum_force` bounds that.
    """

    velocity_gain: float = field(default=4000.0, kw_only=True)
    """
    Extra stiffness per metre per second of approach speed, in N s/m^2.

    Damps the simulated impact so the press settles instead of ringing.
    """

    maximum_force: float = field(default=50.0, kw_only=True)
    """
    Largest force this surface pushes the whole contact face back with, in N.
    """

    face_samples: int = field(default=5, kw_only=True)
    """
    How many points across each side of the tool's contact face the surface is sampled
    at.

    One point cannot tell a tool lying flat from one resting on an edge, and a real
    sensor tells them apart by the torque. A grid of springs produces that torque.
    """

    _penetration: float = field(init=False, default=0.0)
    """
    Mean penetration measured on the previous reading, in m.
    """

    _tool_face: np.ndarray = field(init=False, repr=False)
    """
    The sampled points of the tool's contact face, homogeneous, in the tool's own frame.
    """

    def __post_init__(self):
        if self.surface_height is None:
            self.surface_height = (
                self.surface.collision.as_bounding_box_collection_in_frame(
                    self.world.root
                )
                .bounding_box()
                .max_z
            )
        box = self.tool.collision.as_bounding_box_collection_in_frame(
            self.tool
        ).bounding_box()
        # The face that meets the surface is the one the tool's own z axis points out of,
        # which is how a tool is held: the axis away from the hand faces what it works on.
        across = np.linspace(box.min_x, box.max_x, self.face_samples)
        along = np.linspace(box.min_y, box.max_y, self.face_samples)
        grid_x, grid_y = np.meshgrid(across, along)
        self._tool_face = np.stack(
            [
                grid_x.ravel(),
                grid_y.ravel(),
                np.full(grid_x.size, box.max_z),
                np.ones(grid_x.size),
            ]
        )

    def over_the_surface(self, points: np.ndarray) -> np.ndarray:
        """
        :param points: Points in the world frame, as columns.
        :return: Which of them stand over the surface rather than beside it, where
            there is nothing to touch.
        """
        box = self.surface.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()
        return (
            (box.min_x <= points[0])
            & (points[0] <= box.max_x)
            & (box.min_y <= points[1])
            & (points[1] <= box.max_y)
        )

    def contact_wrench(
        self, sensor: ForceTorqueSensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        world_T_tool = self.world.compute_forward_kinematics_np(
            self.world.root, self.tool
        )
        world_T_sensor = self.world.compute_forward_kinematics_np(
            self.world.root, sensor.root
        )
        points = (world_T_tool @ self._tool_face)[:3]
        penetrations = self.surface_height - points[2]
        touching = (penetrations > 0.0) & self.over_the_surface(points)

        mean_penetration = (
            float(penetrations[touching].sum()) / self._tool_face.shape[1]
        )
        approach_speed = max(
            0.0, (mean_penetration - self._penetration) / self.control_period
        )
        self._penetration = mean_penetration
        if not touching.any():
            return np.zeros(3), np.zeros(3)

        # Each sampled point is its own spring, so the springs under the low edge of a
        # tilted tool push harder than those under the high edge, and the difference is
        # what the sensor reads as torque.
        per_point = (
            self.stiffness + self.velocity_gain * approach_speed
        ) / self._tool_face.shape[1]
        magnitudes = penetrations * touching * per_point
        # The cap bounds what the whole face is pushed with. Scaling the springs down
        # together rather than clipping each keeps the lopsidedness that is the torque.
        pushed_with = magnitudes.sum()
        if pushed_with > self.maximum_force:
            magnitudes = magnitudes * (self.maximum_force / pushed_with)
        world_forces = np.zeros((3, len(magnitudes)))
        world_forces[2] = magnitudes
        levers = points - world_T_sensor[:3, 3][:, None]
        world_force = world_forces.sum(axis=1)
        world_torque = np.cross(levers.T, world_forces.T).sum(axis=0)

        sensor_R_world = world_T_sensor[:3, :3].T
        return sensor_R_world @ world_force, sensor_R_world @ world_torque


# %% the demonstrations


@dataclass
class WipingDemonstration(RobotDemonstration, ABC):
    """
    A demonstration that wipes a surface with a sponge held in one hand.
    """

    technique: WipingTechnique = WipingTechnique.SHEAR
    """
    How the tool covers the surface: lanes, a spiral or a shear.
    """

    sponge_name: str = "wipe_sponge"
    """
    Name of the sponge body bolted to the gripper.
    """

    sponge_scale: Scale = field(default_factory=lambda: Scale(0.08, 0.055, 0.03))
    """
    Size of the sponge, in m.
    """

    sponge_protrusion: float = 0.02
    """
    How far the sponge's pressing face stands beyond the fingertips holding it, in m.

    The fingers must stay clear of the surface while the press yields into it, and what
    is left of the sponge's thickness is what the fingers hold it by.
    """

    press: Press = field(
        default_factory=lambda: Press(
            force=Vector3(z=15.0),
            # Only along the surface normal: the tool feels friction while it travels,
            # and a press free to yield to that drifts against its own direction of
            # travel until it stalls.
            travel=Vector3(x=0.0, y=0.0, z=0.02),
        )
    )
    """
    How hard the tool is held against the surface, and how it may yield to get there.
    """

    simulated_control_period: float = 1 / 50
    """
    Time between two simulated control cycles, in s, matching the rate coraplex
    simulates at.
    """

    @classproperty
    def sponge_contact(cls) -> ContactParameters:
        """
        What the sponge's surface does when it is pressed onto something: soft over a
        centimetre of squash and only moderately grippy, as a damp sponge is.

        .. warning:: The resolution time leaves the pad about as stiff as
            :attr:`SimulatedContactWrench.stiffness`. A tenth of it makes the pad a
            rigid block the press can only chase: half the wipe out of contact,
            touching down at 132 N instead of 12 N.
        """
        return ContactParameters(
            friction=ContactFriction(sliding=0.4),
            stiffness=ContactStiffness(time_constant=timedelta(milliseconds=500)),
            impedance=ContactImpedance(minimum=0.4, maximum=0.7, width=0.01),
        )

    def wiping_arm(self, world: World) -> Arm:
        """
        :param world: World holding the robot.
        :return: The arm that holds the sponge, the left one where the robot has two.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        return robot.get_left_arm_if_specified() or robot.all_arms[0]

    @property
    def unmodelled_tool_length(self) -> float:
        """
        How much further from the hand the sponge really sits than the robot description
        says, in m. Zero when the description is complete.

        .. warning:: The press can only yield by
            :attr:`AdmittanceCartesianTrajectory.maximum_offset`, so a description short
            by more than that drives the tool through the surface.
        """
        return 0.0

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(self.sponge_name)

    def attach_sponge(self, world: World) -> Sponge:
        """
        Bolt the sponge into the gripper's fingers and annotate it as a tool.

        :param world: World holding the robot.
        :return: The sponge, whose frame follows the wipe path.
        """
        end_effector = self.wiping_arm(world).end_effector
        sponge_body = Body(
            name=PrefixedName(self.sponge_name),
            collision=ShapeCollection([Box(scale=self.sponge_scale)]),
            visual=ShapeCollection([Box(scale=self.sponge_scale)]),
        )
        self.sponge_contact.apply_to([sponge_body])
        sponge = Sponge(root=sponge_body)
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=end_effector.tool_frame,
                    child=sponge_body,
                    parent_T_connection_expression=self.held_sponge_pose(end_effector),
                )
            )
            world.add_semantic_annotations([sponge])
        return sponge

    def held_sponge_pose(
        self, end_effector: HasTwoFingers
    ) -> HomogeneousTransformationMatrix:
        """
        :param end_effector: The gripper holding the sponge.
        :return: Where the sponge sits in that gripper's tool frame: between the
            fingertips, turned so the fingers close on its narrow side, and sticking
            :attr:`sponge_protrusion` past their tips.
        """
        fingertips = end_effector.fingertips
        grasp = fingertips.tool_P_grasp.to_np().flatten()
        closing_axis = fingertips.closing_axis.to_np().flatten()
        tips_end_at = max(
            tip.collision.as_bounding_box_collection_in_frame(end_effector.tool_frame)
            .bounding_box()
            .max_z
            for tip in fingertips.both
        )
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(grasp[0]),
            y=float(grasp[1]),
            z=tips_end_at
            + self.sponge_protrusion
            - self.sponge_scale.z / 2.0
            + self.unmodelled_tool_length,
            # Turning the sponge's own y axis onto the closing axis lays its long side
            # across the fingers.
            yaw=float(np.arctan2(-closing_axis[0], closing_axis[1])),
        )

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

    def world_inputs(self, world: World) -> List[InputSynchronizer]:
        """
        :param world: World the demonstration acts on.
        :return: What writes the force/torque reading into the world while the plan
            runs: nothing on the real robot, which follows its sensor's own topic, the
            physics wherever a plan is performed in any, and a contact model otherwise.

        The source is attached to the sensor it writes, so the sensor states where its
        readings come from rather than the loop alone knowing.
        """
        if self.execution_type is ExecutionType.REAL:
            return []
        tool = world.get_body_by_name(self.sponge_name)
        if self.simulation is None:
            pressed_with = self.contact_model(world)
        else:
            pressed_with = PhysicallySimulatedContactWrench(
                world=world, tool=tool, simulation=self.simulation
            )
        sensor = ForceTorqueSensor.for_tip(world, tool)
        sensor.use_source(pressed_with)
        return [sensor.source]

    @abstractmethod
    def contact_model(self, world: World) -> SimulatedContactWrench:
        """
        :param world: World the demonstration acts on.
        :return: The surface this demonstration presses on, modelled for a run that
            resolves no contacts of its own.
        """


@dataclass
class WipeTableDemonstration(WipingDemonstration):
    """
    The HSRB wipes the kitchen's dining table, one side at a time.
    """

    ros_node_name: ClassVar[str] = "wipe_table_demo_node"

    sides: Optional[int] = None
    """
    How many of the usable sides to wipe from, or ``None`` for all of them.

    One placement reaches under half of this table, so covering it means driving round
    it. Set this to shorten a demonstration.
    """

    approach_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=2.0, y=5.705, z=0.0
        )
    )
    """
    Where the robot is put down in the kitchen, in front of the dining table.
    """

    standing_clearance: float = 0.05
    """
    Gap left between the base and the table edge when standing at a side, in m.
    """

    waypoint_stride: int = 10
    """
    Keep every Nth point of the generated tool path.

    The path is sampled at 0.4 mm, far finer than a wipe needs, and the trajectory
    advances at most a couple of waypoints per control cycle, so the raw density alone
    decides how long the motion takes. Thinning it costs no coverage.
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
                    odom_T_robot_start=self.approach_pose,
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
                    arm=self.wiping_arm(world),
                    tool=sponge,
                    surface=table.root,
                    technique=self.technique,
                    pointer_stride=self.waypoint_stride,
                    skip_unreachable=True,
                    press=self.press,
                )
            )
        return sequential(steps, context=context)

    def contact_model(self, world: World) -> SimulatedContactWrench:
        return SimulatedContactWrench(
            world=world,
            tool=world.get_body_by_name(self.sponge_name),
            surface=world.get_semantic_annotations_by_type(DiningTable)[0].root,
            control_period=self.simulated_control_period,
        )

    def standing_poses(self, world: World, table: DiningTable) -> List[Pose]:
        """
        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :return: Where the base can stand to wipe, one pose per usable side, just clear
            of the edge and facing the table, nearest side first.
        """
        usable = [
            side
            for side in self.sides_of(world, table)
            if self.has_room_for_the_base(world, table, side.pose)
            and self.arm_reaches_the_middle(world, table, side.pose)
        ]
        robot_position = world.compute_forward_kinematics_np(
            world.root, self.wiping_arm(world)._robot.root
        )[:2, 3]
        return [
            side.pose
            for side in sorted(usable, key=lambda side: side.preference(robot_position))
        ]

    def sides_of(self, world: World, table: DiningTable) -> List[TableSideStanding]:
        """
        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :return: Where the base would stand at each of the table's four sides.
        """
        body = table.root
        extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        world_T_table = world.compute_forward_kinematics_np(world.root, body)
        table_yaw = math.atan2(world_T_table[1, 0], world_T_table[0, 0])
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        standoff = robot.mobile_base.base_radius + self.standing_clearance

        standings = []
        for side in TableSide:
            normal_x, normal_y = side.value
            world_P_base = world_T_table @ np.array(
                [
                    normal_x * (0.5 * extent.depth + standoff),
                    normal_y * (0.5 * extent.width + standoff),
                    0.0,
                    1.0,
                ]
            )
            standings.append(
                TableSideStanding(
                    pose=Pose.from_xyz_rpy(
                        x=float(world_P_base[0]),
                        y=float(world_P_base[1]),
                        z=0.0,
                        yaw=table_yaw + math.atan2(-normal_y, -normal_x),
                        reference_frame=world.root,
                    ),
                    across=extent.depth if normal_x else extent.width,
                )
            )
        return standings

    def arm_reaches_the_middle(
        self, world: World, table: DiningTable, pose: Pose
    ) -> bool:
        """
        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :param pose: Where the base would stand.
        :return: Whether the arm reaches the middle of the table top from there.

        The wipe is laid out around the middle of the surface, so a side the arm cannot
        reach the middle from covers none of it.
        """
        body = table.root
        middle = world.compute_forward_kinematics_np(world.root, body)[:2, 3]
        standing = pose.to_np()[:2, 3]
        return bool(
            np.linalg.norm(middle - standing) <= self.wiping_arm(world).maximum_reach()
        )

    def has_room_for_the_base(
        self, world: World, table: DiningTable, pose: Pose
    ) -> bool:
        """
        Whether the base fits where a side would be wiped from.

        The table itself is ignored, since :attr:`standing_clearance` already holds the
        base off its edge.

        :param world: World holding the table and whatever surrounds it.
        :param table: The table being wiped.
        :param pose: Where the base would stand.
        :return: Whether that spot is clear.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        base = robot.mobile_base
        base_box = base.bounding_box
        clearance = base.base_radius + self.standing_clearance
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


def main(execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    """
    WipeTableDemonstration(used_robot=HSRB, execution_type=execution_type).run()


if __name__ == "__main__":
    # Only what the command line names, so the default lives once, in main.
    main(*(ExecutionType[name] for name in sys.argv[1:2]))
