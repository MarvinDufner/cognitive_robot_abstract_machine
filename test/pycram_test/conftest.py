import os
from copy import deepcopy

import pytest
import rclpy

from pycram.datastructures.dataclasses import Context

from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.procthor.procthor_semantic_annotations import Milk
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Scale, Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    VizMarkerPublisher(world, node)  # Initialize the publisher
    yield
    rclpy.shutdown()


@pytest.fixture(scope="function")
def mutable_model_world():
    world = setup_world()
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


@pytest.fixture(scope="session")
def pr2_world_setup():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "pycram",
        "resources",
        "robots",
    )
    pr2 = os.path.join(urdf_dir, "pr2_calibrated_with_ft.urdf")
    pr2_parser = URDFParser.from_file(file_path=pr2)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    return world_with_pr2


@pytest.fixture(scope="session")
def whole_apartment_world(pr2_world_setup):
    pr2_sem_world = deepcopy(pr2_world_setup)
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    apartment_world.merge_world(pr2_sem_world)
    apartment_world.merge_world(milk_world)
    apartment_world.merge_world(cereal_world)

    apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
        TransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        )
    )
    apartment_world.get_body_by_name(
        "breakfast_cereal.stl"
    ).parent_connection.origin = TransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.05, reference_frame=apartment_world.root
    )
    milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    robot_view = PR2.from_world(apartment_world)
    return apartment_world, robot_view, Context(apartment_world, robot_view)


@pytest.fixture
def immutable_model_world(whole_apartment_world):
    world, pr2, _ = whole_apartment_world
    state = deepcopy(world.state.data)
    yield world, pr2, Context(world, pr2)
    world.state.data = state


@pytest.fixture(scope="session")
def simple_pr2_world_setup(pr2_world_setup):
    world = deepcopy(pr2_world_setup)
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    world.merge_world(milk_world)
    with world.modify_world():
        world.get_body_by_name("milk.stl").parent_connection.origin = (
            TransformationMatrix.from_xyz_rpy(0.8, 0, 1.05)
        )

        box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )
        connection = FixedConnection(
            parent=world.root,
            child=box,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                1, 0, 0.5, reference_frame=world.root
            ),
        )
        world.add_connection(connection)
    robot_view = PR2.from_world(world)
    return world, robot_view, Context(world, robot_view)


@pytest.fixture
def simple_pr2_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    state = deepcopy(world.state.data)
    yield world, robot_view, context
    world.state.data = state
