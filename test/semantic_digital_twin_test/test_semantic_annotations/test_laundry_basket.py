"""
Tests for the laundry basket annotation and the grasp on its laundry it records.
"""

from __future__ import annotations

import numpy
import pytest

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import LaundryGraspNotPerceived, NoClothInWorld
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cloth,
    Container,
    LaundryBasket,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% fixtures


@pytest.fixture
def laundry_grasp_pose(laundry_basket: LaundryBasket) -> Pose:
    """
    :return: A grasp approaching the laundry straight down, in the basket's world frame.
    """
    return Pose.from_xyz_rpy(
        x=0.6,
        y=0.1,
        z=0.35,
        pitch=numpy.pi / 2,
        reference_frame=laundry_basket._world.root,
    )


def add_cloth(
    *, world: World, name: str, top: HomogeneousTransformationMatrix
) -> Cloth:
    """
    :param world: The world to add the cloth to.
    :param name: The name of the cloth's body.
    :param top: Where the middle of the cloth's top lies, in the world frame.
    :return: A folded T-shirt, a flat box of cloth.
    """
    scale = Scale(0.2, 0.2, 0.05)
    body = BodySpecification.box(
        name, scale, origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-scale.z / 2)
    ).spawn(world, parent_T_self=top)
    with world.modify_world():
        cloth = Cloth(root=body)
        world.add_semantic_annotation(cloth)
    return cloth


@pytest.fixture
def perceived_laundry_basket(
    laundry_basket: LaundryBasket, laundry_grasp_pose: Pose
) -> LaundryBasket:
    """
    :return: The laundry basket, with the grasp perception found on its laundry.
    """
    with laundry_basket._world.modify_world():
        laundry_basket.update_laundry_grasp_pose(laundry_grasp_pose)
    return laundry_basket


@pytest.fixture
def grasped_cloth(
    perceived_laundry_basket: LaundryBasket, laundry_grasp_pose: Pose
) -> Cloth:
    """
    :return: The cloth whose top the perceived grasp lies on.
    """
    return add_cloth(
        world=perceived_laundry_basket._world,
        name="grasped_t_shirt",
        top=HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=laundry_grasp_pose.position
        ),
    )


# %% what a laundry basket is


def test_a_laundry_basket_is_perceivable(laundry_basket: LaundryBasket) -> None:
    """
    A laundry basket can be perceived.
    """
    assert isinstance(laundry_basket, IsPerceivable)


def test_a_laundry_basket_is_a_container(laundry_basket: LaundryBasket) -> None:
    """
    A laundry basket holds laundry and has an opening it goes in and out through.
    """
    assert isinstance(laundry_basket, Container)


def test_a_new_laundry_basket_has_no_laundry_grasp(
    laundry_basket: LaundryBasket,
) -> None:
    """
    Before perception has looked into it, a laundry basket knows no grasp on its
    laundry.
    """
    assert laundry_basket.laundry_grasp_pose is None


# %% recording the grasp on the laundry


def test_the_laundry_basket_records_the_grasp_on_its_laundry(
    laundry_basket: LaundryBasket, laundry_grasp_pose: Pose
) -> None:
    """
    The grasp perception found on the laundry is recorded on the basket.
    """
    with laundry_basket._world.modify_world():
        laundry_basket.update_laundry_grasp_pose(laundry_grasp_pose)

    assert laundry_basket.laundry_grasp_pose is laundry_grasp_pose


def test_the_laundry_basket_keeps_its_laundry_grasp_after_a_json_roundtrip(
    laundry_basket: LaundryBasket, laundry_grasp_pose: Pose
) -> None:
    """
    The grasp travels with the basket when it is serialized, as worlds do to
    synchronize.
    """
    with laundry_basket._world.modify_world():
        laundry_basket.update_laundry_grasp_pose(laundry_grasp_pose)
    serialized = laundry_basket.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(laundry_basket._world)

    restored = LaundryBasket.from_json(data=serialized, **tracker.create_kwargs())

    assert numpy.allclose(
        restored.laundry_grasp_pose.to_np(), laundry_grasp_pose.to_np()
    )


# %% taking hold of the laundry


def test_the_laundry_is_not_grasped_before_it_was_perceived(
    laundry_basket: LaundryBasket,
) -> None:
    """
    Without a grasp perception found, there is nowhere to take hold of the laundry.
    """
    with pytest.raises(LaundryGraspNotPerceived):
        laundry_basket.grasp_on_laundry(pinch_depth=0.02)


def test_the_laundry_is_not_grasped_in_a_world_without_cloth(
    perceived_laundry_basket: LaundryBasket,
) -> None:
    """
    A grasp has to be on some cloth.
    """
    with pytest.raises(NoClothInWorld):
        perceived_laundry_basket.grasp_on_laundry(pinch_depth=0.02)


def test_the_laundry_is_grasped_on_the_cloth_the_grasp_lies_on(
    perceived_laundry_basket: LaundryBasket, grasped_cloth: Cloth
) -> None:
    """
    Of several cloths, the grasp is on the one whose surface the perceived grasp lies
    on.
    """
    add_cloth(
        world=perceived_laundry_basket._world,
        name="other_t_shirt",
        top=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.6, y=-0.2, z=0.3),
    )

    grasp = perceived_laundry_basket.grasp_on_laundry(pinch_depth=0.02)

    assert grasp.graspable is grasped_cloth


def test_the_laundry_is_grasped_on_a_cloth_of_unknown_shape(
    perceived_laundry_basket: LaundryBasket,
) -> None:
    """
    A cloth without collision geometry, whose shape perception does not know, can be
    taken hold of.
    """
    world = perceived_laundry_basket._world
    body = BodySpecification(name="laundry").spawn(
        world, parent=perceived_laundry_basket.root
    )
    with world.modify_world():
        cloth = Cloth(root=body)
        world.add_semantic_annotation(cloth)

    grasp = perceived_laundry_basket.grasp_on_laundry(pinch_depth=0.02)

    assert grasp.graspable is cloth


def test_a_cloth_of_unknown_shape_is_taken_hold_of_where_its_origin_lies_nearest(
    perceived_laundry_basket: LaundryBasket, laundry_grasp_pose: Pose
) -> None:
    """
    Of several cloths without collision geometry, the grasp is on the one whose origin
    lies nearest to the perceived grasp.
    """
    world = perceived_laundry_basket._world
    cloths = []
    for name, distance in [("far_laundry", 0.3), ("near_laundry", 0.1)]:
        body = BodySpecification(name=name).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=laundry_grasp_pose.position
            )
            @ HomogeneousTransformationMatrix.from_xyz_rpy(y=distance),
        )
        with world.modify_world():
            cloths.append(Cloth(root=body))
            world.add_semantic_annotation(cloths[-1])
    _, near_cloth = cloths

    grasp = perceived_laundry_basket.grasp_on_laundry(pinch_depth=0.02)

    assert grasp.graspable is near_cloth


def test_the_fingers_close_the_pinch_depth_past_the_perceived_grasp(
    perceived_laundry_basket: LaundryBasket,
    grasped_cloth: Cloth,
    laundry_grasp_pose: Pose,
) -> None:
    """
    The grasp moves along its approach, into the laundry, by the pinch depth.
    """
    pinch_depth = 0.02

    grasp = perceived_laundry_basket.grasp_on_laundry(pinch_depth=pinch_depth)

    approach = laundry_grasp_pose.rotation_matrix.x_vector()
    expected_position = laundry_grasp_pose.position + approach * pinch_depth
    distance = grasp.world_T_grasp.position.euclidean_distance(expected_position)
    assert float(distance) == pytest.approx(0.0)


def test_a_turned_grasp_closes_along_the_same_line_from_the_other_side(
    perceived_laundry_basket: LaundryBasket, grasped_cloth: Cloth
) -> None:
    """
    Turned half a turn about its approach, the grasp keeps its approach and its closing
    line, but closes the other way along it.
    """
    grasp = perceived_laundry_basket.grasp_on_laundry(pinch_depth=0.02)

    turned = perceived_laundry_basket.grasp_on_laundry(
        pinch_depth=0.02, turned_about_approach=True
    )

    world_T_grasp = grasp.world_T_grasp
    world_T_turned = turned.world_T_grasp
    grasp_rotation = world_T_grasp.rotation_matrix
    turned_rotation = world_T_turned.rotation_matrix
    assert float(
        world_T_turned.position.euclidean_distance(world_T_grasp.position)
    ) == pytest.approx(0.0)
    assert float(
        turned_rotation.x_vector().dot(grasp_rotation.x_vector())
    ) == pytest.approx(1.0)
    assert float(
        turned_rotation.y_vector().dot(grasp_rotation.y_vector())
    ) == pytest.approx(-1.0)
