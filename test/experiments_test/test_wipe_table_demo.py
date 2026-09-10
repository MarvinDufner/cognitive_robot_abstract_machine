import math

import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType
from experiments.wipe_table_demo.demo import (
    CONTACT_STIFFNESS,
    SimulatedContactWrench,
    OUTWARD_NORMALS,
    PRESS_FORCE,
    SPONGE_NAME,
    STANDING_CLEARANCE,
    WipeTableDemonstration,
)
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DiningTable,
    Sponge,
)


def populated_demonstration(**arguments):
    demonstration = WipeTableDemonstration(used_robot=HSRB, **arguments)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return demonstration, world


# %% scene


def test_the_scene_counts_as_populated_once_the_sponge_is_attached():
    demonstration = WipeTableDemonstration(used_robot=HSRB)
    world = demonstration.build_simulated_world()

    assert not demonstration.is_scene_populated(world)
    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_populating_brings_in_the_table_and_the_sponge():
    _, world = populated_demonstration()

    assert world.get_semantic_annotations_by_type(DiningTable)
    assert world.get_semantic_annotations_by_type(Sponge)
    assert world.get_body_by_name(SPONGE_NAME).has_collision()


def test_the_sponge_hangs_off_the_gripper():
    """
    It has to move with the hand: the wipe drives the sponge's own frame along the tool
    path.
    """
    _, world = populated_demonstration()

    sponge = world.get_body_by_name(SPONGE_NAME)

    assert sponge.parent_connection.parent.name.name == "hand_gripper_tool_frame"


def test_the_sponge_is_measured_by_the_wrist_sensor():
    """
    The admittance resolves its sensor from the tool, so the sponge has to sit below the
    wrist sensor on the kinematic chain.
    """
    _, world = populated_demonstration()

    sensor = ForceTorqueSensor.for_tip(world, world.get_body_by_name(SPONGE_NAME))

    assert sensor.root.name.name == "wrist_ft_sensor_frame"


# %% where the robot stands


def test_only_sides_the_base_fits_at_are_offered():
    """
    The kitchen counter runs along one side of the dining table, so the robot cannot
    stand there however close that side happens to be.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]

    poses = demonstration.standing_poses(world, table)

    assert 0 < len(poses) < len(OUTWARD_NORMALS)
    for pose in poses:
        assert demonstration.has_room_for_the_base(world, table, pose)


def test_the_shallow_side_is_wiped_from_first():
    """
    The arm has to reach across the table, so the side with less table in front of it is
    worth more than the side that happens to be nearest.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    body = table.root
    extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
    world_T_table = world.compute_forward_kinematics_np(world.root, body)

    first = demonstration.standing_poses(world, table)[0]

    table_P_base = np.linalg.inv(world_T_table) @ np.append(first.to_np()[:3, 3], 1.0)
    stands_along_x = abs(table_P_base[0]) > abs(table_P_base[1])
    assert stands_along_x == (extent.depth < extent.width)


def test_the_base_stands_clear_of_the_table_edge():
    """
    Standing on the edge would put the base in collision before the arm ever moves.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    body = table.root
    extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
    world_T_table = world.compute_forward_kinematics_np(world.root, body)
    robot = world.get_semantic_annotations_by_type(HSRB)[0]
    expected = robot.mobile_base.footprint_radius + STANDING_CLEARANCE

    for pose in demonstration.standing_poses(world, table):
        table_P_base = np.linalg.inv(world_T_table) @ np.append(
            pose.to_np()[:3, 3], 1.0
        )
        outside_x = abs(table_P_base[0]) - 0.5 * extent.depth
        outside_y = abs(table_P_base[1]) - 0.5 * extent.width
        assert max(outside_x, outside_y) == pytest.approx(expected, abs=1e-6)


def test_the_robot_faces_the_table_from_wherever_it_stands():
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    table_centre = world.compute_forward_kinematics_np(world.root, table.root)[:2, 3]

    for pose in demonstration.standing_poses(world, table):
        pose_np = pose.to_np()
        heading = math.atan2(pose_np[1, 0], pose_np[0, 0])
        towards_table = table_centre - pose_np[:2, 3]
        assert math.cos(heading - math.atan2(*towards_table[::-1])) > 0.99


# %% force control


def test_the_real_robot_reads_its_own_sensor():
    """The simulated contact model stands in for the sensor, so a real run must not get
    one: the controller follows the sensor's own topic instead."""
    demonstration, world = populated_demonstration(execution_type=ExecutionType.REAL)

    assert demonstration.world_inputs(world) == []


def test_a_simulated_run_supplies_its_own_contact():
    demonstration, world = populated_demonstration()

    inputs = demonstration.world_inputs(world)

    assert [type(reading) for reading in inputs] == [SimulatedContactWrench]


def test_the_contact_reading_leaves_the_kinematic_state_alone():
    """
    A wrench is a measurement, not a degree of freedom.

    Reporting a write would recompute the forward kinematics once per reading.
    """
    demonstration, world = populated_demonstration()
    contact = demonstration.world_inputs(world)[0]

    assert contact.apply() is False


def test_the_wipe_presses_into_the_surface_and_travels_across_it():
    """
    End to end: the admittance finds the table without being told its exact height, then
    holds the press while the tool covers ground.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    surface_height = demonstration.surface_height(world, table)
    start = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(SPONGE_NAME)
    )[:3, 3].copy()

    world = demonstration.run()

    sponge = world.get_body_by_name(SPONGE_NAME)
    end = world.compute_forward_kinematics_np(world.root, sponge)[:3, 3]
    assert np.linalg.norm(end[:2] - start[:2]) > 0.1
    # The press settles where the surface pushes back as hard as it is pressed.
    np.testing.assert_allclose(
        end[2] - surface_height, -PRESS_FORCE / CONTACT_STIFFNESS, atol=2e-3
    )
