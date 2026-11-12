"""
The HSRB wipes a kitchen table, regulating how hard it presses with its wrist
force/torque sensor, and driving the base along the table to reach what the arm cannot.

The wipe is coraplex's :class:`~coraplex.robot_plans.actions.composite.tool_based.
WipingAction`; giving it a ``press`` turns its geometric path into a compliant one.
"""

from __future__ import annotations

import math
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta

import numpy as np

from coraplex.datastructures.enums import WipingTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.costmaps import Costmap, OccupancyCostmap
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    PhysicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.composite.tool_based import (
    LowerUntilContactAction,
    WipingAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from cramph.composites import Sequence
from giskardpy.motion_statechart.tasks.admittance_tasks import Press
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import ExecutorExtension
from cramph.statechart import Statechart
from giskardpy.motion_control import WorldStateInputSynchronization
from semantic_digital_twin.adapters.sensors.force_torque import (
    PhysicallySimulatedContactWrench,
)
from semantic_digital_twin.input_synchronization import (
    WorldStateInputs,
)
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
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
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
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Scale
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

    def preference(self, robot_position: np.ndarray) -> tuple[float, float]:
        """
        :param robot_position: Where the robot stands now, in the world frame.
        :return: What orders this side against the others: shallow sides first, since
            the arm has to reach across the table, and among equals the one the robot
            is already closest to.
        """
        return self.across, float(
            np.linalg.norm(self.pose.to_np()[:2, 3] - robot_position)
        )


@dataclass(frozen=True)
class WipedPatch:
    """
    One part of a surface, covered in a single wiping motion.
    """

    centre: Pose
    """
    The middle of the patch, on the surface, with the lanes along its own x axis.
    """

    length: float
    """
    Length of the patch along the lanes, in m.
    """

    width: float
    """
    Width of the patch across the lanes, in m.
    """


# %% the demonstrations


@dataclass
class WipingDemonstration(RobotDemonstration, ABC):
    """
    A demonstration that wipes a surface with a sponge held in one hand.
    """

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor
    """
    Physics by default, since the press is regulated against a measured contact force
    and only a physics engine resolves one, see
    :attr:`~coraplex.demonstrations.RobotDemonstration.executor_type`.

    .. warning:: A kinematically simulated run feels nothing, so its descent never
        reaches contact and its press never yields.
    """

    technique: WipingTechnique = WipingTechnique.SPREAD
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
            # Light, because the mass over the damping is phase lag the loop does not
            # need: 10 kg costs a third of the contact.
            mass=Vector3(x=2.0, y=2.0, z=2.0),
            # The press chases the surface at the force asked for divided by this,
            # so 5 cm/s here, and it is the only one of these numbers that measurably
            # decides how hard: 15 N peaks at 32 N with 100, 20 N with 150, 17 N here.
            damping=Vector3(x=300.0, y=300.0, z=300.0),
            # Only along the surface normal: the tool feels friction while it travels,
            # and a press free to yield to that drifts against its own direction of
            # travel until it stalls.
            maximum_offset=Vector3(x=0.0, y=0.0, z=0.02),
            # About half of the 50 ms the arm takes to reach a commanded height.
            lead_time=0.03,
        )
    )
    """
    How hard the tool is held against the surface, and how it may yield to get there.
    """

    wipe_speed: float = 0.045
    """
    How fast the tool travels while wiping, in m/s.

    The tendency to be thrown off a crest grows with the square of the speed, so this is
    the speed at which the press keeps contact over an uneven surface as well as on a
    flat one.
    """

    sponge_resolution_time: timedelta = timedelta(milliseconds=500)
    """
    How long the simulation takes to resolve a contact with the sponge's pad, which is
    how soft the pad is.

    .. warning:: A tenth of this makes the pad a rigid block the press can only chase:
        half the wipe out of contact, touching down at 132 N instead of 12 N.
    """

    @property
    def sponge_contact(self) -> ContactParameters:
        """
        :return: What the sponge's surface does when it is pressed onto something: soft
            over a centimetre of squash and only moderately grippy, as a damp sponge is.
        """
        return ContactParameters(
            friction=ContactFriction(sliding=0.4),
            stiffness=ContactStiffness(time_constant=self.sponge_resolution_time),
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

    approach_height: float = 0.08
    """
    How far above the surface the tool is brought before it is lowered onto it, in m.
    """

    turning_height: float = 0.30
    """
    How far above the surface the tool is turned to face it, in m.

    The tool reaches a commanded position before it reaches a commanded orientation, so
    a tool that turns on its way down meets the surface edge first. It turns up here
    instead and holds that orientation for the rest of the plan.
    """

    touchdown_force: float = 5.0
    """
    Force magnitude that ends the descent onto the surface, in N: above what the sensor
    reports out of contact, far below what the press holds.
    """

    descent_velocity: float = 0.02
    """
    How fast the tool is lowered onto the surface, in m/s.

    Peak contact force grows with the speed the tool arrives at, and the pad meets a
    crest on a line rather than on its whole face, so arriving slowly is what keeps a
    touchdown off a crest from spiking.
    """

    descent_overshoot: float = 0.02
    """
    How far below the surface the descent aims, in m. Contact ends the descent, so the
    goal has to lie past the surface, and close enough to bound a descent that feels
    nothing.
    """

    @abstractmethod
    def touchdown_pose(self, world: World) -> Pose:
        """
        :param world: World the demonstration acts on.
        :return: Where on the surface the tool is set down before it starts wiping, in
            the world frame.
        """

    @abstractmethod
    def wiped_surface(self, world: World) -> Body:
        """
        :param world: World the demonstration acts on.
        :return: The body whose top the tool is held against.
        """

    def surface_height(self, world: World) -> float:
        """
        :param world: World the demonstration acts on.
        :return: Height of the wiped surface in the world frame, in m.
        """
        return float(
            self.wiped_surface(world)
            .collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .max_z
        )

    def build_approach(
        self, context: StatechartContext, patch: Pose
    ) -> list[MoveManipulatorAction]:
        """
        :param context: Plan context holding the robot.
        :param patch: Where the tool is set down, in the world frame.
        :return: Turning the tool onto the patch well above it, then coming straight
            down to the height the descent starts from.

        .. warning:: The approach has to point the tool down. From the opposite
            orientation the wipe's alignment would have to flip the tool through the one
            pose where it cannot tell which way to turn.
        """
        end_effector = self.wiping_arm(context.world).end_effector
        return [
            MoveManipulatorAction(
                target_pose=self.pose_over_patch(context, patch, height),
                end_effector=end_effector,
                # Collision avoidance stays on: the approach holds the tool over the
                # surface without touching it, and only the descent may touch.
                allow_gripper_collision=False,
            )
            for height in (self.turning_height, self.approach_height)
        ]

    @staticmethod
    def pose_over_patch(context: StatechartContext, patch: Pose, height: float) -> Pose:
        """
        :param context: Plan context holding the world.
        :param patch: Where the tool is set down, in the world frame.
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

    def build_touchdown(
        self, context: StatechartContext, patch: Pose
    ) -> LowerUntilContactAction:
        """
        :param context: Plan context holding the robot.
        :param patch: Where the tool is set down, in the world frame.
        :return: Lowering the tool until the sensor feels the surface, so where the
            surface really is is measured rather than assumed.
        """
        world = context.world
        below_the_surface = deepcopy(patch)
        below_the_surface.z = patch.z - self.descent_overshoot
        return LowerUntilContactAction(
            goal_pose=below_the_surface,
            arm=self.wiping_arm(world),
            tool=world.get_semantic_annotations_by_type(Sponge)[0],
            contact_force=self.touchdown_force,
            descent_velocity=self.descent_velocity,
        )

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Give the plan the robot in ``world``.
        """
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_executor_extensions(self, world: World) -> list[ExecutorExtension]:
        """
        Write the force/torque reading into the world before every control cycle, for a
        run whose readings nothing else publishes.
        """
        inputs = self.wrench_inputs(world)
        if not inputs.synchronizers:
            return []
        return [WorldStateInputSynchronization(inputs=inputs)]

    def wrench_inputs(self, world: World) -> WorldStateInputs:
        """
        :param world: World the demonstration acts on.
        :return: What writes the force/torque reading into the world while the plan
            runs: the physics wherever a plan is performed in any, and nothing
            otherwise, since a real sensor publishes its own topic and nothing resolves
            a contact for a run that only moves the kinematics.

        The source is attached to the sensor it writes, so the sensor states where its
        readings come from rather than the loop alone knowing.
        """
        inputs = WorldStateInputs(world=world)
        if self.simulation is None:
            return inputs
        tool = world.get_body_by_name(self.sponge_name)
        sensor = ForceTorqueSensor.for_tip(world, tool)
        sensor.use_source(
            PhysicallySimulatedContactWrench(
                world=world, tool=tool, simulation=self.simulation
            )
        )
        inputs.add_input(sensor.source)
        return inputs


@dataclass
class RidgedSurface(WipingDemonstration, ABC):
    """
    A wiping demonstration whose surface carries ridges across the wiped patch, so the
    tool has to follow a surface nothing told it about.

    .. warning:: Only a physically simulated run resolves the ridges. A real run would
        wipe the bare surface while the plan believes the ridges are there.
    """

    wave_amplitude: float = 0.02
    """
    How far the crests stand above the surface, in m.
    """

    wave_length: float = 0.12
    """
    Distance between two crests, in m, along the sweep.
    """

    wave_radius: float = 0.06
    """
    Radius of the ridges, in m: the curvature the tool is thrown off by at a crest.
    """

    ridge_length: float = 0.6
    """
    Length of each ridge, in m, across the lanes it is laid under: wider than the patch
    they cover, and no wider than the surface carrying it.
    """

    ridged_span: float = 0.3
    """
    How far the crests reach along the lanes, in m: the length of the wiped patch, or
    further.

    .. warning:: Crests beyond the surface stand unsupported in mid air, and a mobile
        base drives into them, so a span wider than the surface is only safe where
        nothing has to pass beside it.
    """

    def populate_scene(self, world: World) -> None:
        """
        Populate the scene as the flat demonstration does, then lay the ridges across
        the wiped patch.
        """
        super().populate_scene(world)
        # Laid out in the frame the lanes are: a crest has to lie across them, or the
        # tool runs along one instead of riding over it.
        world_T_patch = self.touchdown_pose(world).to_np()
        # Rounded up, so the crests cover the span rather than stopping short of it.
        crests = math.ceil(0.5 * self.ridged_span / self.wave_length)
        # Bolted to the surface rather than to the world root: a ridge is a bump on
        # that surface, so it travels with it and whatever ignores the surface, such as
        # a base looking for room to stand, ignores the ridges with it.
        surface = self.wiped_surface(world)
        surface_T_world = np.linalg.inv(
            world.compute_forward_kinematics_np(world.root, surface)
        )
        with world.modify_world():
            for index in range(-crests, crests + 1):
                patch_T_ridge = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=index * self.wave_length,
                    y=0.0,
                    # Sunk into the surface, so only the crest stands out of it and the
                    # tool rides a wave rather than a step.
                    z=self.wave_amplitude - self.wave_radius,
                    roll=np.pi / 2.0,
                )
                world.add_connection(
                    FixedConnection(
                        parent=surface,
                        child=self.ridge(f"ridge_{index + crests}"),
                        parent_T_connection_expression=HomogeneousTransformationMatrix(
                            surface_T_world @ world_T_patch @ patch_T_ridge.to_np(),
                            reference_frame=surface,
                        ),
                    )
                )

    def ridge(self, name: str) -> Body:
        """
        :param name: Name the ridge is added under.
        :return: One ridge, lying across the surface.
        """
        return Body(
            name=PrefixedName(name),
            collision=ShapeCollection([self.ridge_shape()]),
            visual=ShapeCollection([self.ridge_shape()]),
        )

    def ridge_shape(self) -> Cylinder:
        """
        :return: The shape of one ridge, as its own instance, since a body's collision
            and its visual each own theirs.
        """
        return Cylinder(width=2.0 * self.wave_radius, height=self.ridge_length)


@dataclass
class WipeTableDemonstration(WipingDemonstration):
    """
    The HSRB wipes the kitchen's dining table, one side at a time.
    """

    ros_node_name: str = "wipe_table_demo_node"
    """
    Name of the node this demonstration talks to ROS under.
    """

    placements: int | None = None
    """
    How many of the places the base can stand to wipe from, or ``None`` for all of them.

    One placement covers only what the arm reaches from it, so covering the table means
    driving along it. Set this to shorten a demonstration.
    """

    turning_height: float = 0.15
    """
    How far above the table the tool is turned to face it, in m.

    Lower than a long-armed robot would need: the HSRB holds the tool out across the
    table, and the higher it has to hold it there the sooner it runs out of reach.
    """

    approach_height: float = 0.05
    """
    How far above the table the tool is brought before it is lowered onto it, in m.
    """

    approach_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=3.45, y=5.705, z=0.0, yaw=math.pi
        )
    )
    """
    Where the robot is localized in the kitchen, facing the dining table.

    .. warning:: Clear of the cooking counter along the far side of the table, which
        the base used to be put down inside of. A kinematic run tolerates that; a
        physics engine resolves it.
    """

    standing_map_resolution: float = 0.02
    """
    Edge length of a cell of the map of where the base fits, in m.
    """

    standing_map_cells: int = 200
    """
    How many cells that map holds along each side, which at the resolution above
    reaches 2 m around the table.
    """

    standing_clearance: float = 0.05
    """
    Gap left between the base and the table edge when standing at a side, in m.
    """

    flat_reach: float = 0.49
    """
    How far from its base the arm can hold the pad flat on a table of this height, in m.

    .. note:: Measured by sweeping the arm with the lift solved for the table's height.
        A bound on the tool frame's distance in any posture at all reads 0.69 m; holding
        the pad within eight degrees of flat up there costs a third of that.
    """

    simulation_time_per_action: timedelta | None = timedelta(seconds=40)
    """
    Long enough for one lap of lanes over a patch, which takes tens of seconds at the
    speed the tool is wiped at rather than the seconds a single reaching motion takes.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        An empty map; the kitchen is added by :meth:`populate_scene`.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        """
        The robot, localized where it first wipes from.

        .. note:: The pose goes on the odometry frame rather than on the drive, so that
            a physics engine starts the robot where the plan has it: a drive's own pose
            is a joint position, and a physics engine builds a body at the pose of its
            parent frame.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            world_T_odom=self.approach_pose,
        )

    def build_simulated_world(self) -> World:
        """
        Spawns the robot before the scene is populated, because the sponge is bolted
        into its gripper, see
        :meth:`~coraplex.demonstrations.RobotDemonstration.build_simulated_world`.
        """
        world = self.scene_specification().create_world()
        self.robot_specification().spawn(world)
        self.populate_scene(world)
        return world

    def populate_scene(self, world: World) -> None:
        """
        Add the kitchen and bolt the sponge into the gripper.
        """
        if not world.get_semantic_annotations_by_type(DiningTable):
            kitchen = KitchenEnvironment().create_world()
            with world.modify_world():
                # Bolted to the world root. Merging without a connection leaves the
                # kitchen on six free degrees of freedom, which a physics engine reads
                # as one unanchored rigid body and drops under gravity.
                world.merge_world(
                    kitchen,
                    FixedConnection(parent=world.root, child=kitchen.root),
                )
        self.attach_sponge(world)

    def build_statechart(self, context: StatechartContext) -> Statechart:
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
        for standing_pose in self.standing_poses(world, table)[: self.placements]:
            patch = self.reachable_patch(world, standing_pose)
            steps.append(NavigateAction(target_location=standing_pose))
            # No pose-based approach: the descent aligns the tool from the tool's own
            # alignment pairs, while a commanded pose would have to state an
            # orientation convention, and the one Tracy uses is not the HSRB's.
            steps.append(self.build_touchdown(context, patch.centre))
            steps.append(
                WipingAction(
                    arm=self.wiping_arm(world),
                    tool=sponge,
                    target_pose=patch.centre,
                    length=patch.length,
                    width=patch.width,
                    technique=self.technique,
                    press=self.press,
                    max_linear_velocity=self.wipe_speed,
                )
            )
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

    def reachable_patch(self, world: World, standing: Pose) -> WipedPatch:
        """
        :param world: World holding the table.
        :param standing: Where the base stands while the tool presses.
        :return: The largest part of the table top the arm can hold the pad flat on from
            there, with the lanes running along the edge the robot stands at.

        The base stays where it stands while the tool presses, so the arm sweeps a disc
        of :attr:`flat_reach` around it and the deepest rectangle that fits inside that
        disc is the one whose far corners sit on its rim.
        """
        footprint = (
            self.wiped_surface(world)
            .collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
        )
        base = standing.to_np()[:2, 3]
        heading = math.atan2(standing.to_np()[1, 0], standing.to_np()[0, 0])
        inwards = np.array([math.cos(heading), math.sin(heading)])
        along = np.array([-inwards[1], inwards[0]])
        corners = np.array(
            [
                [footprint.min_x, footprint.min_y],
                [footprint.min_x, footprint.max_y],
                [footprint.max_x, footprint.min_y],
                [footprint.max_x, footprint.max_y],
            ]
        )
        inward_reach = (corners - base) @ inwards
        lateral_reach = (corners - base) @ along
        near_edge = float(inward_reach.min())
        # The far corners of the rectangle sit on the rim of the reach, which is where
        # the area it encloses is largest.
        farthest = float(
            np.clip(
                (near_edge + math.sqrt(near_edge**2 + 8.0 * self.flat_reach**2)) / 4.0,
                near_edge,
                self.flat_reach,
            )
        )
        depth = min(farthest - near_edge, float(inward_reach.max()) - near_edge)
        half_width = min(
            math.sqrt(max(0.0, self.flat_reach**2 - farthest**2)),
            float(lateral_reach.max()),
            -float(lateral_reach.min()),
        )
        centre = base + inwards * (near_edge + 0.5 * depth)
        return WipedPatch(
            centre=Pose.from_xyz_rpy(
                x=float(centre[0]),
                y=float(centre[1]),
                z=self.surface_height(world),
                # The lanes run along the edge, which is the long way across the patch.
                yaw=heading + 0.5 * math.pi,
                reference_frame=world.root,
            ),
            length=2.0 * half_width,
            width=depth,
        )

    def touchdown_pose(self, world: World) -> Pose:
        """
        :param world: World holding the table and the robot.
        :return: The middle of the patch wiped from the first place the robot stands, so
            the tool is set down where it starts wiping rather than somewhere it then
            has to travel from.
        """
        table = world.get_semantic_annotations_by_type(DiningTable)[0]
        return self.reachable_patch(world, self.standing_poses(world, table)[0]).centre

    def surface_yaw(self, world: World) -> float:
        """
        :param world: World holding the table.
        :return: Which way the table's own x axis points in the world, in rad.
        """
        world_T_surface = world.compute_forward_kinematics_np(
            world.root, self.wiped_surface(world)
        )
        return math.atan2(world_T_surface[1, 0], world_T_surface[0, 0])

    def wiped_surface(self, world: World) -> Body:
        return world.get_semantic_annotations_by_type(DiningTable)[0].root

    def standing_poses(self, world: World, table: DiningTable) -> list[Pose]:
        """
        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :return: Where the base can stand to wipe, spaced along every usable side so
            that the patches reached from them meet, just clear of the edge and facing
            the table, nearest side first.
        """
        room = self.room_for_the_base(world)
        usable = [
            side for side in self.sides_of(world, table) if room.allows(side.pose)
        ]
        robot_position = world.compute_forward_kinematics_np(
            world.root, self.wiping_arm(world)._robot.root
        )[:2, 3]
        return [
            pose
            for side in sorted(usable, key=lambda side: side.preference(robot_position))
            for pose in self.stops_along(world, table, side, room)
        ]

    def stops_along(
        self,
        world: World,
        table: DiningTable,
        side: TableSideStanding,
        room: Costmap,
    ) -> list[Pose]:
        """
        :param world: World holding the table.
        :param table: The table being wiped.
        :param side: The side of the table the base stands at.
        :param room: Where the base fits, see :meth:`room_for_the_base`.
        :return: Where to stand along that side, from the middle outwards, so that the
            patches reached from them cover its length without leaving a strip between.
        """
        body = table.root
        extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        edge_length = extent.width if side.across == extent.depth else extent.depth
        covered = self.reachable_patch(world, side.pose).length
        stops = max(1, math.ceil(edge_length / covered))
        middle = side.pose.to_np()
        heading = math.atan2(middle[1, 0], middle[0, 0])
        along = np.array([-math.sin(heading), math.cos(heading)])
        spaced = [
            Pose.from_xyz_rpy(
                x=float(middle[0, 3] + along[0] * offset),
                y=float(middle[1, 3] + along[1] * offset),
                z=0.0,
                yaw=heading,
                reference_frame=world.root,
            )
            for offset in (
                (index - 0.5 * (stops - 1)) * edge_length / stops
                for index in range(stops)
            )
        ]
        return [stop for stop in spaced if room.allows(stop)]

    def sides_of(self, world: World, table: DiningTable) -> list[TableSideStanding]:
        """
        :param world: World holding the table and the robot.
        :param table: The table being wiped.
        :return: Where the base would stand at each of the table's four sides.
        """
        body = table.root
        extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        world_T_table = world.compute_forward_kinematics_np(world.root, body)
        table_yaw = self.surface_yaw(world)
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

    def room_for_the_base(self, world: World) -> Costmap:
        """
        :param world: World holding the table and whatever surrounds it.
        :return: Where the base fits, as a map of the floor around the table that is
            clear wherever an obstacle is further than the base's radius plus
            :attr:`standing_clearance` away.

        .. note:: The table counts as an obstacle like any other, so the map already
            holds the base off the edge it wipes from.
        """
        table = world.get_semantic_annotations_by_type(DiningTable)[0]
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        standing_height = deepcopy(table.root.global_pose)
        standing_height.z = 0.0
        return OccupancyCostmap(
            resolution=self.standing_map_resolution,
            width=self.standing_map_cells,
            height=self.standing_map_cells,
            world=world,
            distance_to_obstacle=(
                robot.mobile_base.base_radius + self.standing_clearance
            ),
            robot_view=robot,
            origin=standing_height,
        )


def main(
    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor,
    technique: WipingTechnique = WipeTableDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan.
    :param technique: How the tool covers the table.
    """
    WipeTableDemonstration(
        used_robot=HSRB, executor_type=executor_type, technique=technique
    ).run()


if __name__ == "__main__":
    main()
