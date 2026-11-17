"""
Tests for the laundry basket annotation and the grasp on its laundry it records.
"""

from __future__ import annotations

import numpy
import pytest

from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Container,
    LaundryBasket,
)
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture
def laundry_basket() -> LaundryBasket:
    """
    :return: A laundry basket in a world of its own.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        return LaundryBasket.create_with_new_body_in_world(
            name="laundry_basket", world=world
        )


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
