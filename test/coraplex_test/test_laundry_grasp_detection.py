"""
Tests for finding where to take hold of the laundry in a basket, read from the world
model the way the robot's camera sees it.
"""

from __future__ import annotations

import numpy
import pytest

from coraplex.exceptions import MotionDidNotFinish
from coraplex.perception import LaundryGraspDetection
from coraplex.robot_plans.actions.core.navigation import LookAtAction
from cramph.composites import Sequence
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cloth,
    LaundryBasket,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..plan_running import robot_extensions, run_plan

# %% the scene


@pytest.fixture
def hsrb(hsr_world_copy: World) -> HSRB:
    """
    :return: The HSR, standing at the origin of a world of its own.
    """
    [robot] = hsr_world_copy.get_semantic_annotations_by_type(HSRB)
    return robot


@pytest.fixture
def basket(hsrb: HSRB) -> LaundryBasket:
    """
    :return: A laundry basket on the floor in front of the HSR.
    """
    scale = Scale(0.35, 0.5, 0.25)
    world = hsrb._world
    with world.modify_world():
        return LaundryBasket.create_with_new_body_in_world(
            name="laundry_basket",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.75, z=scale.z / 2, reference_frame=world.root
            ),
            scale=scale,
        )


def add_crumpled_t_shirt(
    *, world: World, name: str, middle: tuple[float, float, float], radius: float
) -> Cloth:
    """
    :param world: The world to add the T-shirt to.
    :param name: The name of the T-shirt's body.
    :param middle: Where the middle of the T-shirt lies, in the world frame.
    :param radius: How far the T-shirt reaches from its middle.
    :return: A T-shirt crumpled into a ball of cloth.
    """
    x, y, z = middle
    body = BodySpecification.sphere(name, radius).spawn(
        world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, z=z, reference_frame=world.root
        ),
    )
    with world.modify_world():
        t_shirt = Cloth(root=body)
        world.add_semantic_annotation(t_shirt)
    return t_shirt


def look_for_the_laundry_grasp(*, hsrb: HSRB, basket: LaundryBasket) -> None:
    """
    Let the HSR look into the basket and find where to take hold of its laundry.
    """
    run_plan(
        Sequence(
            nodes=[
                LookAtAction(target=Pose(reference_frame=basket.root)),
                LaundryGraspDetection(basket=basket),
            ]
        ),
        robot_extensions(hsrb),
    )


# %% where the grasp lies


def test_the_grasp_lies_on_top_of_the_highest_laundry_in_the_basket(
    hsrb: HSRB, basket: LaundryBasket
) -> None:
    """
    The camera sees past the head it is mounted in, and of the laundry it sees in the
    basket, the grasp lies on the highest point.
    """
    world = hsrb._world
    add_crumpled_t_shirt(world=world, name="low", middle=(0.7, 0.1, 0.06), radius=0.05)
    add_crumpled_t_shirt(world=world, name="high", middle=(0.8, -0.1, 0.1), radius=0.06)

    look_for_the_laundry_grasp(hsrb=hsrb, basket=basket)

    position = basket.laundry_grasp_pose.to_np()[:3, 3]
    assert position == pytest.approx([0.8, -0.1, 0.16], abs=0.01)


def test_laundry_outside_the_basket_is_not_grasped(
    hsrb: HSRB, basket: LaundryBasket
) -> None:
    """
    Laundry lying higher beside the basket does not belong to it.
    """
    world = hsrb._world
    add_crumpled_t_shirt(
        world=world, name="inside", middle=(0.75, 0.0, 0.06), radius=0.05
    )
    add_crumpled_t_shirt(
        world=world, name="beside", middle=(0.75, 0.45, 0.3), radius=0.05
    )

    look_for_the_laundry_grasp(hsrb=hsrb, basket=basket)

    position = basket.laundry_grasp_pose.to_np()[:3, 3]
    assert position == pytest.approx([0.75, 0.0, 0.11], abs=0.01)


def test_the_grasp_comes_straight_down_and_closes_along_the_image(
    hsrb: HSRB, basket: LaundryBasket
) -> None:
    """
    The grasp approaches along its x-axis and closes along the camera image's horizontal
    axis, laid flat.
    """
    add_crumpled_t_shirt(
        world=hsrb._world, name="t_shirt", middle=(0.75, 0.0, 0.06), radius=0.05
    )

    look_for_the_laundry_grasp(hsrb=hsrb, basket=basket)

    rotation = basket.laundry_grasp_pose.to_np()[:3, :3]
    image_x_axis = hsrb.get_default_camera().root.global_transform.to_np()[:3, 0]
    flat_image_x_axis = numpy.array([image_x_axis[0], image_x_axis[1], 0.0])
    assert rotation[:, 0] == pytest.approx([0.0, 0.0, -1.0])
    assert rotation[:, 1] == pytest.approx(
        flat_image_x_axis / numpy.linalg.norm(flat_image_x_axis)
    )


def test_an_empty_basket_offers_no_grasp(hsrb: HSRB, basket: LaundryBasket) -> None:
    """
    Seeing no laundry in the basket fails the detection and records no grasp.
    """
    with pytest.raises(MotionDidNotFinish):
        look_for_the_laundry_grasp(hsrb=hsrb, basket=basket)

    assert basket.laundry_grasp_pose is None
