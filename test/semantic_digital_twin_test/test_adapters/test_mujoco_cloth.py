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
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.garments import TShirt
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.air import Air
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
    :return: A stepped simulation of the T-shirt above the table, with the integrator
        MuJoCo simulates elastic cloth with.
    """
    simulation = MujocoSim(
        world=t_shirt_above_table.world,
        headless=True,
        step_size=2e-3,
        integrator=mujoco.mjtIntegrator.mjINT_DISCRETE,
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


@pytest.mark.parametrize("mass", [0.15, 1.5 * MujocoBuilder.minimum_mass])
def test_a_cloth_sheet_spreads_its_mass_evenly_over_its_vertices(
    tmp_path: Path, triangle: trimesh.Trimesh, mass: float
) -> None:
    """
    Every vertex of the cloth weighs an equal share of its mass, even a share lighter
    than the builder's minimum mass.
    """
    cloth = ClothSheet(name="cloth", surface=triangle, mass=mass)

    model = build_model_of(cloth, tmp_path)

    vertex_bodies = model.flex_vertbodyid[: len(cloth.surface.vertices)]
    assert model.body_mass[vertex_bodies] == pytest.approx(cloth.vertex_mass)


def test_a_t_shirt_weighs_its_mass(tmp_path: Path) -> None:
    """
    The vertices of a T-shirt as light as RGBench's are heavier than the builder's
    minimum mass, so the simulated T-shirt weighs what it is given.
    """
    t_shirt = ClothSheet(
        name="t_shirt",
        surface=GarmentFile.T_SHIRT.create_surface(HomogeneousTransformationMatrix()),
        mass=0.255,
    )

    model = build_model_of(t_shirt, tmp_path)

    vertex_bodies = model.flex_vertbodyid[: len(t_shirt.surface.vertices)]
    assert model.body_mass[vertex_bodies].sum() == pytest.approx(t_shirt.mass)


@pytest.mark.skipif(
    not PR2.is_description_installed(), reason="The PR2's description is not installed"
)
def test_a_light_cloth_leaves_the_mass_of_a_robots_bodies_as_it_is(
    tmp_path: Path,
) -> None:
    """
    A T-shirt whose vertices are lighter than the builder's minimum mass leaves every
    body of a PR2, among them its massless jointed fingertip frames, as heavy as it is
    without the T-shirt.
    """

    def body_masses(with_t_shirt: bool) -> dict[str, float]:
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        RobotSpecification(PR2).spawn(world)
        if with_t_shirt:
            world.add_simulator_property(
                ClothSheet(
                    name="t_shirt",
                    surface=GarmentFile.T_SHIRT.create_surface(
                        HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, z=0.8)
                    ),
                    mass=0.255,
                )
            )
        scene = tmp_path / f"with_t_shirt_{with_t_shirt}.xml"
        MujocoBuilder().build_world(world=world, file_path=str(scene))
        model = mujoco.MjModel.from_xml_path(str(scene))
        vertex_bodies = set(model.flex_vertbodyid.tolist())
        return {
            model.body(body_id).name: float(model.body_mass[body_id])
            for body_id in range(model.nbody)
            if body_id not in vertex_bodies
        }

    assert body_masses(with_t_shirt=True) == pytest.approx(
        body_masses(with_t_shirt=False)
    )


def test_the_air_drags_on_a_t_shirt_across_its_whole_area(tmp_path: Path) -> None:
    """
    Each vertex of the T-shirt carries a massless sphere that touches nothing, for the
    air to drag on, and together their cross-sections cover the T-shirt's area.
    """
    t_shirt = ClothSheet(
        name="t_shirt",
        surface=GarmentFile.T_SHIRT.create_surface(HomogeneousTransformationMatrix()),
        mass=0.255,
    )

    model = build_model_of(t_shirt, tmp_path)

    vertex_bodies = model.flex_vertbodyid[: len(t_shirt.surface.vertices)]
    drag_geoms = numpy.flatnonzero(numpy.isin(model.geom_bodyid, vertex_bodies))
    assert drag_geoms.size == len(t_shirt.surface.vertices)
    assert (model.geom_type[drag_geoms] == mujoco.mjtGeom.mjGEOM_SPHERE).all()
    assert numpy.pi * (model.geom_size[drag_geoms, 0] ** 2).sum() == pytest.approx(
        t_shirt.surface.area
    )
    assert (model.geom_contype[drag_geoms] == 0).all()
    assert (model.geom_conaffinity[drag_geoms] == 0).all()
    assert model.body_mass[vertex_bodies].sum() == pytest.approx(t_shirt.mass)


def test_air_fills_the_simulation_with_its_density_and_viscosity(
    tmp_path: Path, triangle: trimesh.Trimesh
) -> None:
    """
    A world in air is simulated in a fluid of the air's density and viscosity.
    """
    world = world_holding(ClothSheet(name="cloth", surface=triangle, mass=0.15))
    air = Air()
    world.add_simulator_property(air)

    MujocoBuilder().build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))

    assert model.opt.density == pytest.approx(air.density)
    assert model.opt.viscosity == pytest.approx(air.viscosity)


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


def test_a_t_shirt_hanging_from_its_neckline_does_not_stretch_its_edges() -> None:
    """
    A T-shirt hanging from the vertices around the middle of its neckline carries its
    whole weight on the few edges next to them, which still stretch by less than a tenth
    of their length.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        hand = Body(name=PrefixedName("hand"))
        world.add_connection(
            FixedConnection(
                parent=root,
                child=hand,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.5, reference_frame=root
                ),
            )
        )
    # the T-shirt's file lays it out with the middle of its neckline at the origin and
    # its hem along the x-axis, which the pitch turns downwards
    t_shirt = ClothSheet(
        name="t_shirt",
        surface=GarmentFile.T_SHIRT.create_surface(
            HomogeneousTransformationMatrix.from_xyz_rpy(z=1.5, pitch=numpy.pi / 2)
        ),
        mass=0.255,
        thickness=3e-3,
    )
    world.add_simulator_property(t_shirt)
    simulation = MujocoSim(
        world=world,
        headless=True,
        step_size=2e-3,
        integrator=mujoco.mjtIntegrator.mjINT_DISCRETE,
    )
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(milliseconds=2))
        pinned = simulation.simulator.pin_flex_vertices(
            flex_name=t_shirt.name,
            body_name=hand.name.name,
            body_P_centre=numpy.zeros(3),
            radius=0.05,
        ).result
        simulation.step_simulation(timedelta(seconds=1))
        positions = simulation.simulator.get_flex_vertex_positions(
            flex_name=t_shirt.name
        ).result
    finally:
        simulation.stop_simulation()

    edges = t_shirt.surface.edges_unique
    edge_lengths = numpy.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1
    )
    assert pinned.size > 0
    assert (edge_lengths / t_shirt.surface.edges_unique_length).max() < 1.1


def test_air_stops_a_t_shirt_from_swinging() -> None:
    """
    A T-shirt let fall from the middle of its neckline, where it is held, swings less
    than half as far in air two seconds later as it does without air.
    """

    def swing_after_two_seconds(in_air: bool) -> float:
        world = World()
        with world.modify_world():
            root = Body(name=PrefixedName("root"))
            world.add_body(root)
            hand = Body(name=PrefixedName("hand"))
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=hand,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=1.5, reference_frame=root
                    ),
                )
            )
        # the T-shirt's file lays it out with the middle of its neckline at the origin
        t_shirt = ClothSheet(
            name="t_shirt",
            surface=GarmentFile.T_SHIRT.create_surface(
                HomogeneousTransformationMatrix.from_xyz_rpy(z=1.5)
            ),
            mass=0.255,
            thickness=3e-3,
        )
        world.add_simulator_property(t_shirt)
        if in_air:
            world.add_simulator_property(Air())
        simulation = MujocoSim(
            world=world,
            headless=True,
            step_size=4e-3,
            integrator=mujoco.mjtIntegrator.mjINT_DISCRETE,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(timedelta(milliseconds=4))
            simulation.simulator.pin_flex_vertices(
                flex_name=t_shirt.name,
                body_name=hand.name.name,
                body_P_centre=numpy.zeros(3),
                radius=0.05,
            )
            simulation.step_simulation(timedelta(seconds=1.5))
            before = simulation.simulator.get_flex_vertex_positions(
                flex_name=t_shirt.name
            ).result
            simulation.step_simulation(timedelta(seconds=0.5))
            after = simulation.simulator.get_flex_vertex_positions(
                flex_name=t_shirt.name
            ).result
        finally:
            simulation.stop_simulation()
        return float(numpy.linalg.norm(after - before, axis=1).max())

    assert (
        swing_after_two_seconds(in_air=True) < swing_after_two_seconds(in_air=False) / 2
    )
