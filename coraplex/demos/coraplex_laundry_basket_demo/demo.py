"""
The HSR drives to a laundry basket on the kitchen environment's living room floor and
lifts the top T-shirt out in MuJoCo, by a grasp assumed on its highest point while the
simulation has no perception. The T-shirts are rigid stand-ins, see
:class:`CrumpledTShirt`. Run it from this folder::

    python demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random

import mujoco
import trimesh

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence, StatementNode
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode, SucceedsOnObservingTrue
from cramph.statechart import Statechart
from krrood.entity_query_language.factories import a, variable_from
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.pipeline.mesh_decomposition.vhacd import (
    VHACDMeshDecomposer,
)
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cloth,
    Floor,
    LaundryBasket,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cloth import GarmentFile
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.inertial_properties import (
    InertiaTensor,
    Inertial,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% crumpling a garment


@dataclass(kw_only=True)
class Fold:
    """
    A fold laying the part of a garment beyond a straight crease over the rest of it.

    Every layer bends around an axis above the crease on a half circle, one axis shared
    by the layers it lies no further above than :attr:`largest_bend_radius`, so those on
    top end up inside the fold and the part laid over lands above them.
    """

    garment_T_crease: HomogeneousTransformationMatrix
    """
    The crease in the frame of the garment, whose z-axis points up: its x-axis runs
    along the crease and its y-axis points into the part that is laid over.
    """

    clearance: float
    """
    How far above the garment's highest point the shared axis lies, in metres.
    """

    largest_bend_radius: float
    """
    The widest a layer bends, in metres, so that a layer lying far below the shared axis
    drapes over instead of standing up in a loop.
    """

    def lay_over(self, *, surface: trimesh.Trimesh) -> None:
        """
        Fold the garment's surface in place.

        :param surface: The garment's surface, in the garment's frame.
        """
        surface.apply_transform(self.garment_T_crease.inverse().to_np())
        shared_axis_height = surface.bounds[1][2] + self.clearance
        folded_vertices = []
        for x, y, z in surface.vertices.tolist():
            if y <= 0.0:
                folded_vertices.append([x, y, z])
                continue
            bend_radius = min(shared_axis_height - z, self.largest_bend_radius)
            axis_height = z + bend_radius
            bend_angle = y / bend_radius
            if bend_angle < math.pi:
                folded_vertices.append(
                    [
                        x,
                        bend_radius * math.sin(bend_angle),
                        axis_height - bend_radius * math.cos(bend_angle),
                    ]
                )
            else:
                folded_vertices.append(
                    [x, math.pi * bend_radius - y, axis_height + bend_radius]
                )
        surface.vertices = folded_vertices
        surface.apply_transform(self.garment_T_crease.to_np())


@dataclass(kw_only=True)
class WrinkleWave:
    """
    Parallel wrinkles running across a garment.
    """

    direction: float
    """
    The direction across the wrinkles, as the angle from the garment's x-axis, in
    radians.
    """

    spacing: float
    """
    The distance between neighbouring wrinkles, in metres.
    """

    phase: float
    """
    Where along :attr:`direction` the wrinkles lie, as the phase of the wave at the
    garment's origin, in radians.
    """

    def elevation(self, *, x: float, y: float) -> float:
        """
        :param x: The x-coordinate of a point of the garment, in its frame.
        :param y: The y-coordinate of the point, in the garment's frame.
        :return: How far the wrinkles raise the garment at the point, relative to their
            depth, from -1 to 1.
        """
        distance = x * math.cos(self.direction) + y * math.sin(self.direction)
        return math.sin(2 * math.pi * distance / self.spacing + self.phase)


@dataclass(kw_only=True)
class Crumpling:
    """
    How a garment lying flat is crumpled up: turned, folded over and over across its
    longer side until it fits a footprint, settling under its own weight after each
    fold, then wrinkled, with a tuft pulled up from the middle of its top.

    Its randomness is fixed by :attr:`seed`, so the garment crumples alike in every run.
    """

    seed: int
    """
    The seed of the garment's random turn, creases and wrinkles.
    """

    largest_extent: float
    """
    The most the crumpled garment may span along the x- and the y-axis of its frame, in
    metres.
    """

    crease_skew: float = 0.25
    """
    The most a crease turns away from running straight across the garment's longer side,
    in radians.
    """

    fold_clearance: float = 0.008
    """
    How far above the garment's highest point a fold's layers bend around, in metres.
    """

    largest_bend_radius: float = 0.03
    """
    The widest a layer bends in a fold, in metres.
    """

    settled_height_ratio: float = 0.7
    """
    How much of its height the garment keeps when a fold settles under its own weight.
    """

    wrinkle_depth: float = 0.006
    """
    How far the wrinkles typically raise and lower the garment, in metres.
    """

    wrinkle_spacing: float = 0.08
    """
    The typical distance between neighbouring wrinkles, in metres.

    Each wave of wrinkles is spaced between half and one and a half times as far.
    """

    wrinkle_wave_count: int = 6
    """
    How many waves of wrinkles, each running in its own direction, overlap.
    """

    tuft_height: float = 0.06
    """
    How high the tuft is pulled up above the garment around it, in metres.
    """

    tuft_width: float = 0.02
    """
    How far from its middle the tuft has sunk to :math:`1/e` of its height, in metres.
    """

    def crumple(self, *, surface: trimesh.Trimesh) -> None:
        """
        Crumple a garment's surface in place.

        :param surface: The garment's surface lying flat, its z-axis pointing up. It is
            left with the middle of its footprint and its underside at the origin.
        """
        random_numbers = Random(self.seed)
        surface.apply_transform(
            HomogeneousTransformationMatrix.from_xyz_rpy(
                yaw=random_numbers.uniform(0.0, 2 * math.pi)
            ).to_np()
        )
        self.fold_until_it_fits(surface=surface, random_numbers=random_numbers)
        (min_x, min_y, _), (max_x, max_y, _) = surface.bounds.tolist()
        surface.apply_translation([-(min_x + max_x) / 2, -(min_y + max_y) / 2, 0.0])
        self.wrinkle(surface=surface, random_numbers=random_numbers)
        surface.apply_translation([0.0, 0.0, -surface.bounds[0][2]])

    def fold_until_it_fits(
        self, *, surface: trimesh.Trimesh, random_numbers: Random
    ) -> None:
        """
        Fold the garment's surface in place across its longer side, through the middle
        of its footprint, until it spans no more than :attr:`largest_extent`.

        :param surface: The garment's surface, its z-axis pointing up.
        :param random_numbers: The source of the creases' skews and of which side of
            them is laid over.
        """
        (min_x, min_y, _), (max_x, max_y, _) = surface.bounds.tolist()
        while max(max_x - min_x, max_y - min_y) > self.largest_extent:
            crease_across_x = max_x - min_x >= max_y - min_y
            Fold(
                garment_T_crease=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=(min_x + max_x) / 2,
                    y=(min_y + max_y) / 2,
                    yaw=(math.pi / 2 if crease_across_x else 0.0)
                    + random_numbers.uniform(-self.crease_skew, self.crease_skew)
                    + random_numbers.choice([0.0, math.pi]),
                ),
                clearance=self.fold_clearance,
                largest_bend_radius=self.largest_bend_radius,
            ).lay_over(surface=surface)
            surface.apply_scale([1.0, 1.0, self.settled_height_ratio])
            (min_x, min_y, _), (max_x, max_y, _) = surface.bounds.tolist()

    def wrinkle(self, *, surface: trimesh.Trimesh, random_numbers: Random) -> None:
        """
        Raise and lower the garment's surface in place by waves of wrinkles and by the
        tuft over its origin.

        :param surface: The garment's surface, its z-axis pointing up.
        :param random_numbers: The source of the waves of wrinkles.
        """
        waves = [
            WrinkleWave(
                direction=random_numbers.uniform(0.0, 2 * math.pi),
                spacing=random_numbers.uniform(
                    self.wrinkle_spacing / 2, 3 * self.wrinkle_spacing / 2
                ),
                phase=random_numbers.uniform(0.0, 2 * math.pi),
            )
            for _ in range(self.wrinkle_wave_count)
        ]
        wave_depth = self.wrinkle_depth / math.sqrt(len(waves))
        surface.vertices = [
            [
                x,
                y,
                z
                + wave_depth * sum(wave.elevation(x=x, y=y) for wave in waves)
                + self.tuft_height * math.exp(-(x**2 + y**2) / self.tuft_width**2),
            ]
            for x, y, z in surface.vertices.tolist()
        ]


# %% the grasp assumed in place of perception


@dataclass(eq=False, repr=False, kw_only=True)
class AssumedLaundryGrasp(SucceedsOnObservingTrue, StatechartNode):
    """
    Statechart node that records on a basket a grasp from above on the highest point of
    a T-shirt, where RoboKudo's doing-laundry engine grasps the laundry, in place of
    :class:`~coraplex.perception.LaundryGraspPerception` while the simulation has no
    perception.
    """

    basket: LaundryBasket
    """
    The basket to record the grasp on.
    """

    t_shirt: Body
    """
    The T-shirt on top of the laundry in the basket.
    """

    def on_tick(self) -> ObservationStateValues:
        """
        :meth:`StatechartNode.on_tick`, recording the grasp where the T-shirt lies now
        and observing true.
        """
        world = self.context.world
        [surface] = self.t_shirt.visual.shapes
        x, y, z = max(
            surface.mesh_in_frame(world.root).vertices.tolist(),
            key=lambda vertex: vertex[2],
        )
        with world.modify_world():
            self.basket.update_laundry_grasp_pose(
                Pose.from_xyz_rpy(
                    x=x, y=y, z=z, pitch=math.pi / 2, reference_frame=world.root
                )
            )
        return ObservationStateValues.TRUE


# %% the scene


class SceneBody(StrEnum):
    """
    The names of the bodies the demonstration adds to the kitchen environment.

    .. note:: The world is given the plain names, since a copy of it serialises its
        bodies' names, and this module may not be importable where it is read back.
    """

    LAUNDRY_BASKET = "laundry_basket"
    BOTTOM_LEFT_T_SHIRT = "bottom_left_t_shirt"
    BOTTOM_RIGHT_T_SHIRT = "bottom_right_t_shirt"
    TOP_T_SHIRT = "top_t_shirt"


@dataclass(kw_only=True)
class CrumpledTShirt:
    """
    A rigid stand-in for a T-shirt lying crumpled in the basket, made from
    :attr:`GarmentFile.T_SHIRT`: drawn as its crumpled surface, it collides as the
    convex parts that surface decomposes into.
    """

    name: SceneBody
    """
    The name of the T-shirt's body.
    """

    basket_T_t_shirt: HomogeneousTransformationMatrix
    """
    Where the middle of the T-shirt's underside lies, relative to the middle of the
    basket's underside.
    """

    crumpling: Crumpling
    """
    How the T-shirt is crumpled up.
    """

    garment_scale: float = 0.8
    """
    How large the T-shirt is relative to :attr:`GarmentFile.T_SHIRT`, which is larger
    than most.
    """

    largest_triangle_side: float = 0.02
    """
    The longest side a triangle of the T-shirt's surface has before it is crumpled, in
    metres, short enough for its folds to bend smoothly.
    """

    collision_decomposer: VHACDMeshDecomposer = field(
        default_factory=lambda: VHACDMeshDecomposer(max_convex_hulls=16)
    )
    """
    Decomposes the crumpled T-shirt into the convex parts it collides as.
    """

    mass: float = 0.2
    """
    The mass of the T-shirt, in kilograms.
    """

    color: Color = field(default_factory=Color.BLUE)
    """
    The color the T-shirt is drawn in.
    """

    def body_specification(self) -> BodySpecification:
        """
        :return: The T-shirt as a free body whose frame lies in the middle of its
            underside.
        """
        surface = Mesh(
            filename=str(GarmentFile.T_SHIRT),
            scale=Scale(self.garment_scale, self.garment_scale, self.garment_scale),
        ).mesh
        surface.merge_vertices(merge_tex=True, merge_norm=True)
        surface = surface.subdivide_to_size(max_edge=self.largest_triangle_side)
        self.crumpling.crumple(surface=surface)
        visual = Mesh.from_trimesh(mesh=surface)
        visual.color = self.color
        length, width, height = surface.extents.tolist()
        return BodySpecification(
            name=self.name.value,
            shapes=ShapeCollection(self.collision_decomposer.apply_to_mesh(visual)),
            visual_shapes=ShapeCollection([visual]),
            inertial=Inertial(
                mass=self.mass,
                center_of_mass=Point3(z=height / 2),
                inertia=InertiaTensor.from_values(
                    ixx=self.mass * (width**2 + height**2) / 12,
                    iyy=self.mass * (length**2 + height**2) / 12,
                    izz=self.mass * (length**2 + width**2) / 12,
                    ixy=0.0,
                    ixz=0.0,
                    iyz=0.0,
                ),
            ),
            connection_specification=Connection6DoFSpecification(),
        )


# %% the demonstration


@dataclass(kw_only=True)
class LaundryBasketDemonstration(RobotDemonstration):
    """
    The robot drives to the laundry basket, looks into it and lifts the top T-shirt out
    of the basket by a grasp on the highest point of the laundry.
    """

    used_robot: type[AbstractRobot] = HSRB
    """
    The HSR, the robot of the doing-laundry challenge.
    """

    ros_node_name: str = "hsr_laundry_basket_demo"
    """
    The name of the demonstration's ROS node.
    """

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor
    """
    Performs the plan in MuJoCo, where friction alone holds the lifted T-shirt.
    """

    collision_avoidance: bool = True
    """
    On, since the HSR moves its whole body with its arm and has to keep it clear of the
    basket.
    """

    basket_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-0.06, y=2.75
        )
    )
    """
    Where the middle of the basket's underside stands in the kitchen: against the south
    wall, beside the spot meant for the washing machine, its open side up.
    """

    basket_scale: Scale = field(default_factory=lambda: Scale(0.35, 0.5, 0.25))
    """
    The outer size of the basket.
    """

    robot_start_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.3, y=2.2, yaw=math.pi
        )
    )
    """
    Where the robot starts in the kitchen, facing the south wall.
    """

    standing_distance: float = 0.45
    """
    How far in front of the middle of the basket the robot's base stands to take the
    laundry out, in metres.
    """

    pinch_depth: float = 0.02
    """
    How far past the top of the laundry, along the grasp's approach, the fingers close,
    in metres.
    """

    approach_clearance: float = 0.2
    """
    How far above the T-shirt the gripper opens before it comes down, in metres, enough
    to clear the basket's rim.
    """

    lift_height: float = 0.3
    """
    How far the gripper lifts the T-shirt above where it took hold of it, in metres.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a held T-shirt creep between the
    fingers.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held T-shirt stays where
    the fingers hold it.
    """

    t_shirts: list[CrumpledTShirt] = field(
        default_factory=lambda: [
            CrumpledTShirt(
                name=SceneBody.BOTTOM_LEFT_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.115),
                crumpling=Crumpling(seed=7, largest_extent=0.21, tuft_height=0.0),
            ),
            CrumpledTShirt(
                name=SceneBody.BOTTOM_RIGHT_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.115),
                crumpling=Crumpling(seed=8, largest_extent=0.21, tuft_height=0.0),
                color=Color.GREEN(),
            ),
            CrumpledTShirt(
                name=SceneBody.TOP_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.065),
                crumpling=Crumpling(seed=9, largest_extent=0.3),
                color=Color.RED(),
            ),
        ]
    )
    """
    The T-shirts a simulated run puts into the basket: two side by side on its bottom
    and one lying across both, whose tuft is the highest point of the laundry.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(
            self.used_robot, localization_pose=self.robot_start_pose
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(
            SceneBody.LAUNDRY_BASKET
        )

    def populate_scene(self, world: World) -> None:
        """
        Stand the empty laundry basket against the living room's south wall.
        """
        with world.modify_world():
            LaundryBasket.create_with_new_body_in_world(
                name=SceneBody.LAUNDRY_BASKET.value,
                world=world,
                world_root_T_self=self.basket_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=self.basket_scale.z / 2
                ),
                scale=self.basket_scale,
            )

    def build_simulated_world(self) -> World:
        """
        :meth:`RobotDemonstration.build_simulated_world`, with :attr:`t_shirts` lying in
        the basket, since a real run perceives the laundry instead.
        """
        world = super().build_simulated_world()
        for t_shirt in self.t_shirts:
            body = t_shirt.body_specification().spawn(world)
            world_T_t_shirt = self.basket_pose @ t_shirt.basket_T_t_shirt
            world_T_t_shirt.reference_frame = world.root
            body.parent_connection.origin = world_T_t_shirt
            with world.modify_world():
                world.add_semantic_annotation(Cloth(root=body))
        return world

    def acquire_world(self) -> World:
        """
        :meth:`RobotDemonstration.acquire_world`, with the robot's base allowed to touch
        the floors it rolls on, which its collision avoidance would keep it away from.
        """
        world = super().acquire_world()
        [robot] = world.get_semantic_annotations_by_type(self.used_robot)
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[robot.mobile_base.root],
                    body_group_b=[
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ],
                )
            )
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(
                robot=world.get_semantic_annotations_by_type(self.used_robot)[0]
            )
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park the arm, drive in front of the basket, look into it, assume a grasp on the
        laundry and lift the T-shirt out by it, grounded once it was assumed:

        turned half a turn first, which the HSR reaches without turning its base, and
        closed once the fingers stall on the cloth.
        """
        hsrb = context.require_extension(extension_type=RobotAccess).robot
        [basket] = context.world.get_semantic_annotations_by_type(LaundryBasket)
        [arm] = hsrb.all_arms
        world_T_standing = (
            self.basket_pose
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.standing_distance, yaw=math.pi
            )
        )
        world_T_standing.reference_frame = context.world.root
        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    NavigateAction(target_location=world_T_standing.pose),
                    LookAtAction(target=Pose(reference_frame=basket.root)),
                    AssumedLaundryGrasp(
                        basket=basket,
                        t_shirt=context.world.get_body_by_name(
                            SceneBody.TOP_T_SHIRT.value
                        ),
                    ),
                    StatementNode(
                        statement=a(PickUpAction)(
                            grasp=a(
                                basket.grasp_on_laundry, target_type=GraspCandidate
                            )(
                                pinch_depth=self.pinch_depth,
                                turned_about_approach=variable_from([True, False]),
                            ),
                            arm=arm,
                            approach_clearance=self.approach_clearance,
                            retreat_distance=self.lift_height,
                            tolerate_grasp_stall=True,
                        )
                    ),
                ]
            )
        )
        return statechart

    def build_simulation(self, world: World) -> MujocoSim:
        return MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )


def main(headless: bool = False) -> World:
    """
    Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the T-shirt was lifted in.
    """
    return LaundryBasketDemonstration(headless=headless).run()


if __name__ == "__main__":
    main()
