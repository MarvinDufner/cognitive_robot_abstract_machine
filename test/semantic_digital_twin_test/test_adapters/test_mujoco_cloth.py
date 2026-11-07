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
import trimesh
from typing_extensions import Iterator

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.garments import TShirt
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cloth import ClothSheet, GarmentFile
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
    # the T-shirt's file lays it out from the neckline along x
    t_shirt = ClothSheet(
        name="t_shirt",
        surface=GarmentFile.T_SHIRT.create_surface(
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.4, z=table_top_height + 0.1
            )
        ),
        mass=0.255,
        thickness=3e-3,
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


def world_holding(cloth: ClothSheet) -> World:
    """
    :param cloth: The cloth the world is to hold.
    :return: A world holding nothing but the cloth.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    world.add_simulator_property(cloth)
    return world


def build_model_of(cloth: ClothSheet, directory: Path) -> mujoco.MjModel:
    """
    :param cloth: The cloth to build.
    :param directory: Where the built scene is written.
    :return: The model the builder makes of a world holding nothing but the cloth.
    """
    MujocoBuilder().build_world(
        world=world_holding(cloth), file_path=str(directory / "scene.xml")
    )
    return mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))


@pytest.fixture
def triangle() -> trimesh.Trimesh:
    """
    :return: A single triangle of cloth, small enough to build in an instant.
    """
    return trimesh.Trimesh(
        vertices=[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], faces=[[0, 1, 2]]
    )


def test_a_cloth_sheet_starts_where_its_surface_lies(triangle: trimesh.Trimesh) -> None:
    """
    Every vertex of the simulated cloth starts at its vertex of the surface.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=0.15)
    simulation = MujocoSim(world=world_holding(cloth), headless=True, step_size=2e-3)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(milliseconds=2))
        positions = simulation.simulator.get_flex_vertex_positions(
            flex_name=cloth.name
        ).result
    finally:
        simulation.stop_simulation()

    assert numpy.allclose(positions, cloth.surface.vertices, atol=1e-4)


@pytest.mark.parametrize("mass", [0.15, 0.0015])
def test_a_cloth_sheet_weighs_its_mass_but_every_vertex_at_least_the_minimum_mass(
    tmp_path: Path, triangle: trimesh.Trimesh, mass: float
) -> None:
    """
    The cloth's mass is spread evenly over its vertices, none lighter than the builder's
    minimum mass.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=mass)

    model = build_model_of(cloth, tmp_path)

    vertex_count = len(cloth.surface.vertices)
    vertex_bodies = model.flex_vertbodyid[:vertex_count]
    vertex_mass = max(cloth.mass / vertex_count, MujocoBuilder.minimum_mass)
    assert model.body_mass[vertex_bodies].sum() == pytest.approx(
        vertex_count * vertex_mass
    )


def test_a_cloth_sheet_is_drawn_in_its_color(
    tmp_path: Path, triangle: trimesh.Trimesh
) -> None:
    """
    The simulation draws the cloth in the color the cloth sheet is given.
    """
    cloth = ClothSheet(
        name="cloth",
        surface=triangle,
        mass=0.15,
        color=Color(hue=0.6, saturation=0.5, value=0.9, opacity=0.8),
    )

    model = build_model_of(cloth, tmp_path)

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_rgba[flex_id].tolist() == pytest.approx(
        list(cloth.color.to_rgba())
    )


def test_a_cloth_sheets_edges_are_damped_as_much_as_it_says(
    tmp_path: Path, triangle: trimesh.Trimesh
) -> None:
    """
    The simulation damps the cloth's edges by the cloth sheet's edge damping.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=0.15, edge_damping=0.1)

    model = build_model_of(cloth, tmp_path)

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_edgedamping[flex_id] == pytest.approx(cloth.edge_damping)


def test_a_cloth_sheet_touches_what_it_collides_with_at_its_collision_radius(
    tmp_path: Path, triangle: trimesh.Trimesh
) -> None:
    """
    The simulation lets the cloth touch other objects at the cloth sheet's collision
    radius from its surface.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=0.15, collision_radius=2e-3)

    model = build_model_of(cloth, tmp_path)

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_radius[flex_id] == pytest.approx(cloth.collision_radius)


def test_a_cloth_sheet_does_not_collide_with_itself(
    tmp_path: Path, triangle: trimesh.Trimesh
) -> None:
    """
    The simulation lets the parts of a cloth pass through each other.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=0.15)

    model = build_model_of(cloth, tmp_path)

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_selfcollide[flex_id] == mujoco.mjtFlexSelf.mjFLEXSELF_NONE


def test_a_t_shirt_and_the_cloth_standing_for_it_build_under_one_name(
    tmp_path: Path,
) -> None:
    """
    A T-shirt's root body and the cloth sheet simulating it may share the T-shirt's
    name: the body and the flex are both built, the flex with every vertex.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    t_shirt = TShirt.create_with_new_body_in_world("t_shirt", world)
    cloth = ClothSheet(
        name="t_shirt",
        surface=GarmentFile.T_SHIRT.create_surface(HomogeneousTransformationMatrix()),
        mass=0.255,
        semantic_annotation=t_shirt,
    )
    world.add_simulator_property(cloth)

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))

    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, cloth.name)
    assert model.flex_vertnum[flex_id] == len(cloth.surface.vertices)
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, builder.element_names.of(t_shirt.root)
        )
        != -1
    )


# %% simulating


def test_a_t_shirt_dropped_onto_a_table_lands_on_it_and_keeps_its_extent(
    t_shirt_above_table: TShirtAboveTable, simulation: MujocoSim
) -> None:
    """
    A T-shirt of two layers dropped onto a table lands on its top, sinking into it by
    less than its thickness, and keeps its spread-out extent.

    .. note:: The T-shirt keeps the relief it was laid out with, since that is the shape
        it bends back to, so only its lowest vertices touch the table.
    """
    t_shirt = t_shirt_above_table.t_shirt

    simulation.step_simulation(timedelta(seconds=1))
    positions = simulation.simulator.get_flex_vertex_positions(
        flex_name=t_shirt.name
    ).result

    height_above_table = positions[:, 2] - t_shirt_above_table.table_top_height
    assert -t_shirt.thickness < height_above_table.min() <= t_shirt.collision_radius
    assert numpy.allclose(
        numpy.ptp(positions[:, :2], axis=0), t_shirt.surface.extents[:2], atol=2e-2
    )
