"""
Tests for the HSR lifting the top T-shirt out of a laundry basket in MuJoCo, held by
friction alone.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    LaundryBasket,
)

from ..pytest_environment import runs_in_continuous_integration

pytestmark = [
    pytest.mark.skipif(
        not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
    ),
    pytest.mark.skipif(
        not HSRB.is_description_installed(),
        reason="The description of the HSR is not installed",
    ),
]

# %% the demo, loaded from its folder


@pytest.fixture(scope="module")
def laundry_basket_demo() -> ModuleType:
    """
    The laundry basket demo's ``demo`` module, its folder put on the path for the
    modules it imports.
    """
    folder = (
        Path(__file__).parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_laundry_basket_demo"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(folder))
        specification = importlib.util.spec_from_file_location(
            "laundry_basket_demo", folder / "demo.py"
        )
        module = importlib.util.module_from_spec(specification)
        patch.setitem(sys.modules, specification.name, module)
        specification.loader.exec_module(module)
    return module


# %% lifting the top T-shirt


def test_only_the_top_t_shirt_is_lifted_out_of_the_basket(laundry_basket_demo):
    """
    The highest point of the laundry lies on the top T-shirt, which is lifted by the
    lift height, while the T-shirts below it stay in the basket.
    """
    demonstration = laundry_basket_demo.LaundryBasketDemonstration(headless=True)

    world = demonstration.run()

    for t_shirt in demonstration.t_shirts:
        start = (demonstration.basket_pose @ t_shirt.basket_T_t_shirt).to_np()[:3, 3]
        end = world.get_body_by_name(t_shirt.name).global_transform.to_np()[:3, 3]
        expected_rise = (
            demonstration.lift_height
            if t_shirt.name == laundry_basket_demo.SceneBody.TOP_T_SHIRT
            else 0.0
        )
        assert list(end) == pytest.approx(
            [start[0], start[1], start[2] + expected_rise], abs=0.02
        )


# %% assuming the basket ahead of the robot


@pytest.mark.parametrize("base_yaw", [0.0, 2.0])
def test_basket_is_assumed_ahead_of_the_robot_facing_it(laundry_basket_demo, base_yaw):
    """
    The basket is assumed the configured distance ahead of the robot's base, on the
    floor, turned so that the robot's standing pose in front of it faces the robot.
    """
    robot_start_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=3.0, y=2.0, yaw=base_yaw
    )
    demonstration = laundry_basket_demo.LaundryBasketDemonstration(
        headless=True, robot_start_pose=robot_start_pose
    )
    world = demonstration.build_simulated_world()

    demonstration.place_basket_ahead_of_robot(world=world)

    distance = demonstration.basket_distance_ahead_of_robot
    expected_basket_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=3.0 + distance * math.cos(base_yaw),
        y=2.0 + distance * math.sin(base_yaw),
        yaw=base_yaw + math.pi,
    )
    [basket] = world.get_semantic_annotations_by_type(LaundryBasket)
    assert basket.root.global_transform.to_np() == pytest.approx(
        demonstration.basket_root_pose.to_np()
    )
    assert demonstration.basket_pose.to_np() == pytest.approx(
        expected_basket_pose.to_np()
    )


# %% searching for the basket


def test_the_basket_is_searched_for_as_often_as_configured(laundry_basket_demo):
    """
    Each attempt resets the search, so the search is called off after one reset fewer
    than the attempts it is allowed.
    """
    demonstration = laundry_basket_demo.LaundryBasketDemonstration(
        headless=True, basket_search_attempts=3
    )
    world = demonstration.build_simulated_world()
    [basket] = world.get_semantic_annotations_by_type(LaundryBasket)

    search = demonstration.build_basket_search(basket=basket)

    assert search.stop_retry_monitor.node is search.task
    assert search.stop_retry_monitor.target == demonstration.basket_search_attempts - 1
