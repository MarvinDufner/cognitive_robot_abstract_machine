"""
The HSR drives to a laundry basket on the kitchen environment's living room floor and
lifts the top T-shirt out by the grasp RoboKudo's doing-laundry engine finds, simulated
in MuJoCo. The engine has to be running on the head camera. The T-shirts are rigid
stand-ins, see :class:`TShirtStandIn`. Run it from this folder::

    python demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco

from coraplex.demonstrations import RobotDemonstration
from coraplex.perception import LaundryGraspPerception
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence, StatementNode
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood.entity_query_language.factories import a, variable_from
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
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
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.inertial_properties import (
    InertiaTensor,
    Inertial,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

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
class TShirtStandIn:
    """
    A rigid stand-in for a folded T-shirt lying in the basket: a flat block with a tuft
    of fabric standing up from its top, where a gripper pinches real cloth.
    """

    name: SceneBody
    """
    The name of the T-shirt's body.
    """

    basket_T_t_shirt: HomogeneousTransformationMatrix
    """
    Where the middle of the block's underside lies, relative to the middle of the
    basket's underside.
    """

    block_scale: Scale = field(default_factory=lambda: Scale(0.2, 0.2, 0.05))
    """
    The size of the folded T-shirt.
    """

    block_T_tuft: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where the foot of the tuft stands, relative to the middle of the block's top.
    """

    tuft_radius: float = 0.02
    """
    The radius of the tuft, in metres.
    """

    tuft_height: float = 0.05
    """
    How high the straight part of the tuft stands, below its rounded tip, in metres.
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
        :return: The T-shirt as a free body whose frame lies in the middle of the
            block's underside.
        """
        block_height = self.block_scale.z
        tuft_foot = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=block_height)
            @ self.block_T_tuft
        )
        shapes = ShapeCollection(
            [
                Box(
                    scale=self.block_scale,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=block_height / 2
                    ),
                    color=self.color,
                ),
                Cylinder(
                    width=2 * self.tuft_radius,
                    height=self.tuft_height,
                    origin=tuft_foot
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=self.tuft_height / 2
                    ),
                    color=self.color,
                ),
                Sphere(
                    radius=self.tuft_radius,
                    origin=tuft_foot
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(z=self.tuft_height),
                    color=self.color,
                ),
            ]
        )
        length, width = self.block_scale.x, self.block_scale.y
        return BodySpecification(
            self.name.value,
            shapes,
            inertial=Inertial(
                mass=self.mass,
                center_of_mass=Point3(z=block_height / 2),
                inertia=InertiaTensor.from_values(
                    ixx=self.mass * (width**2 + block_height**2) / 12,
                    iyy=self.mass * (length**2 + block_height**2) / 12,
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
    The robot drives to the laundry basket, looks into it, has RoboKudo find a grasp on
    the laundry and lifts the T-shirt there out of the basket by it.
    """

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "hsr_laundry_basket_demo"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

    collision_avoidance: bool = True

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

    t_shirts: list[TShirtStandIn] = field(
        default_factory=lambda: [
            TShirtStandIn(
                name=SceneBody.BOTTOM_LEFT_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.11, z=0.01
                ),
                block_T_tuft=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.06),
            ),
            TShirtStandIn(
                name=SceneBody.BOTTOM_RIGHT_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.11, z=0.01
                ),
                block_T_tuft=HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.06),
                color=Color.GREEN(),
            ),
            TShirtStandIn(
                name=SceneBody.TOP_T_SHIRT,
                basket_T_t_shirt=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.06),
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
        Park the arm, drive in front of the basket, look into it and lift the T-shirt
        out by the grasp RoboKudo found on the laundry, grounded once it was perceived:

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
                    LaundryGraspPerception(basket=basket),
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
