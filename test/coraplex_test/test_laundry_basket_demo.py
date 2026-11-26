"""
Tests for the HSR lifting the top T-shirt out of a laundry basket in MuJoCo, held by
friction alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from semantic_digital_twin.robots.hsrb import HSRB

from ..pytest_environment import runs_in_continuous_integration

pytestmark = [
    pytest.mark.skip(
        reason="Needs RoboKudo's doing-laundry engine running on the simulated head camera"
    ),
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
