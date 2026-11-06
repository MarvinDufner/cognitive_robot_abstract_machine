"""
Tests for simulating cloth in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import mujoco
import numpy
import pytest
from typing_extensions import Iterator

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cloth import ClothSheet, TShirtPattern
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene


@dataclass
class TShirtAboveTable:
    """
    A T-shirt lying flat a little above a table top, ready to drop onto it.
    """

    world: World
    """
    The world holding the table and the T-shirt.
    """

    t_shirt: ClothSheet
    """
    The T-shirt.
    """

    table_top_height: float
    """
    Height of the table top, in metres.
    """


@pytest.fixture
def t_shirt_above_table() -> TShirtAboveTable:
    """
    :return: A world with a table and a T-shirt lying flat 10 cm above its top.
    """
    table_top_height = 0.7
    table_thickness = 0.04
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        table = Body(name=PrefixedName("table"))
        table.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix(reference_frame=table),
                    scale=Scale(1.2, 1.2, table_thickness),
                )
            ],
            reference_frame=table,
        )
        table.visual = table.collision
        world.add_connection(
            FixedConnection(
                parent=root,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=table_top_height - table_thickness / 2, reference_frame=root
                ),
            )
        )
    t_shirt = ClothSheet(
        name="t_shirt",
        surface=TShirtPattern().create_surface(
            HomogeneousTransformationMatrix.from_xyz_rpy(z=table_top_height + 0.1)
        ),
        mass=0.15,
    )
    world.add_simulator_property(t_shirt)
    return TShirtAboveTable(
        world=world, t_shirt=t_shirt, table_top_height=table_top_height
    )


@pytest.fixture
def simulation(t_shirt_above_table: TShirtAboveTable) -> Iterator[MujocoSim]:
    """
    :return: A stepped simulation of the T-shirt above the table, solved the way cloth
        simulates at an affordable cost.
    """
    simulation = MujocoSim(
        world=t_shirt_above_table.world,
        headless=True,
        step_size=2e-3,
        integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        solver=mujoco.mjtSolver.mjSOL_CG,
    )
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


# %% building


def test_a_cloth_sheet_starts_where_its_surface_lies(
    t_shirt_above_table: TShirtAboveTable, simulation: MujocoSim
) -> None:
    """
    Every vertex of the simulated cloth starts at its vertex of the surface.
    """
    t_shirt = t_shirt_above_table.t_shirt

    simulation.step_simulation(timedelta(milliseconds=2))
    positions = simulation.simulator.get_flex_vertex_positions(
        flex_name=t_shirt.name
    ).result

    assert numpy.allclose(positions, t_shirt.surface.vertices, atol=1e-4)


@pytest.mark.parametrize("mass", [0.15, 1.0])
def test_a_cloth_sheet_weighs_its_mass_but_every_vertex_at_least_the_minimum_mass(
    tmp_path: Path, mass: float
) -> None:
    """
    The cloth's mass is spread evenly over its vertices, none lighter than the builder's
    minimum mass.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    cloth = ClothSheet(
        name="cloth",
        surface=TShirtPattern().create_surface(HomogeneousTransformationMatrix()),
        mass=mass,
    )
    world.add_simulator_property(cloth)
    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    vertex_count = len(cloth.surface.vertices)
    vertex_bodies = model.flex_vertbodyid[:vertex_count]
    vertex_mass = max(cloth.mass / vertex_count, builder.minimum_mass)
    assert model.body_mass[vertex_bodies].sum() == pytest.approx(
        vertex_count * vertex_mass
    )


def test_a_cloth_sheet_is_drawn_in_its_color(tmp_path: Path) -> None:
    """
    The simulation draws the cloth in the color the cloth sheet is given.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    cloth = ClothSheet(
        name="cloth",
        surface=TShirtPattern().create_surface(HomogeneousTransformationMatrix()),
        mass=0.15,
        color=Color(hue=0.6, saturation=0.5, value=0.9, opacity=0.8),
    )
    world.add_simulator_property(cloth)
    MujocoBuilder().build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_rgba[flex_id].tolist() == pytest.approx(
        list(cloth.color.to_rgba())
    )


# %% simulating


def test_a_t_shirt_dropped_onto_a_table_comes_to_rest_flat_on_it(
    t_shirt_above_table: TShirtAboveTable, simulation: MujocoSim
) -> None:
    """
    A T-shirt dropped onto a table rests on its top, within its collision radius, and
    keeps its spread-out extent.
    """
    t_shirt = t_shirt_above_table.t_shirt

    simulation.step_simulation(timedelta(seconds=1))
    positions = simulation.simulator.get_flex_vertex_positions(
        flex_name=t_shirt.name
    ).result

    height_above_table = positions[:, 2] - t_shirt_above_table.table_top_height
    assert numpy.all(height_above_table > 0)
    assert numpy.all(height_above_table <= t_shirt.thickness / 2)
    assert numpy.allclose(
        numpy.ptp(positions[:, :2], axis=0), t_shirt.surface.extents[:2], atol=1e-2
    )
