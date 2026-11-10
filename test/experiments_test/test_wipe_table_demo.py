import math
from inspect import signature

import numpy as np
import pytest

from coraplex.datastructures.enums import WipingTechnique
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from cramph.context import StatechartContext
from experiments.wipe_table_demo.demo import (
    SimulatedContactWrench,
    TableSide,
    WipeTableDemonstration,
    main,
)
from experiments.wipe_table_demo.wavy_demo import WipeWavyTableDemonstration
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DiningTable,
    Sponge,
)


def populated_demonstration(**arguments):
    demonstration = WipeTableDemonstration(used_robot=HSRB, **arguments)
    return demonstration, demonstration.build_simulated_world()


def wiping_actions(demonstration, world) -> list[WipingAction]:
    context = StatechartContext.with_extensions(
        world, demonstration.build_context_extensions(world)
    )
    statechart = demonstration.build_statechart(context)
    return [
        node for node in statechart.nodes if isinstance(node, WipingAction)
    ]


def wiped_points(demonstration, world) -> np.ndarray:
    return np.array(
        [
            point.to_np().flatten()[:3]
            for action in wiping_actions(demonstration, world)
            for point in action._waypoints
        ]
    )


# %% scene


def test_the_scene_counts_as_populated_once_the_sponge_is_attached():
    demonstration = WipeTableDemonstration(used_robot=HSRB)
    world = demonstration.scene_specification().create_world()
    demonstration.robot_specification().spawn(world)

    assert not demonstration.is_scene_populated(world)
    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_populating_brings_in_the_table_and_the_sponge():
    demonstration, world = populated_demonstration()

    assert world.get_semantic_annotations_by_type(DiningTable)
    assert world.get_semantic_annotations_by_type(Sponge)
    assert world.get_body_by_name(demonstration.sponge_name).has_collision()


def test_the_sponge_hangs_off_the_gripper():
    """
    It has to move with the hand: the wipe drives the sponge's own frame along the tool
    path.
    """
    demonstration, world = populated_demonstration()

    sponge = world.get_body_by_name(demonstration.sponge_name)

    assert sponge.parent_connection.parent.name.name == "hand_gripper_tool_frame"


def test_the_sponge_is_measured_by_the_wrist_sensor():
    """
    The admittance resolves its sensor from the tool, so the sponge has to sit below the
    wrist sensor on the kinematic chain.
    """
    demonstration, world = populated_demonstration()

    sensor = ForceTorqueSensor.for_tip(
        world, world.get_body_by_name(demonstration.sponge_name)
    )

    assert sensor.root.name.name == "wrist_ft_sensor_frame"


def test_the_robot_is_localized_rather_than_driven_into_place():
    """
    A drive is no joint a physics engine knows, so a start pose held by the drive never
    reaches one: the robot would be built at the world origin, metres from the table the
    plan is about, while the controller worked from a world that has it at the table.
    """
    demonstration, world = populated_demonstration()
    robot = world.get_semantic_annotations_by_type(HSRB)[0]

    drive = robot.root.parent_connection
    odom = drive.parent.parent_connection

    assert isinstance(drive, OmniDrive)
    assert isinstance(odom, FixedConnection)
    np.testing.assert_allclose(
        world.compute_forward_kinematics_np(world.root, drive.parent),
        demonstration.approach_pose.to_np(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        world.compute_forward_kinematics_np(drive.parent, robot.root),
        np.eye(4),
        atol=1e-9,
    )


# %% where the robot stands


def test_no_place_the_base_does_not_fit_is_offered():
    """
    The kitchen counter runs along one side of the dining table, so the robot is not
    sent to stand there however close that side happens to be.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]

    room = demonstration.room_for_the_base(world)

    poses = demonstration.standing_poses(world, table)

    assert poses
    for pose in poses:
        assert room.allows(pose)
    assert [
        side for side in demonstration.sides_of(world, table) if not room.allows(side.pose)
    ]


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
    expected = robot.mobile_base.base_radius + demonstration.standing_clearance

    for pose in demonstration.standing_poses(world, table):
        table_P_base = np.linalg.inv(world_T_table) @ np.append(
            pose.to_np()[:3, 3], 1.0
        )
        outside_x = abs(table_P_base[0]) - 0.5 * extent.depth
        outside_y = abs(table_P_base[1]) - 0.5 * extent.width
        assert max(outside_x, outside_y) == pytest.approx(expected, abs=1e-6)


def test_the_robot_faces_what_it_wipes_from_wherever_it_stands():
    """
    A stop along an edge faces that edge rather than the middle of the table, and what
    it wipes has to lie ahead of it either way.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]

    for pose in demonstration.standing_poses(world, table):
        standing = pose.to_np()
        heading = math.atan2(standing[1, 0], standing[0, 0])
        ahead = np.array([math.cos(heading), math.sin(heading)])
        patch = demonstration.reachable_patch(world, pose)
        towards_patch = patch.centre.to_np()[:2, 3] - standing[:2, 3]
        assert float(towards_patch @ ahead) == pytest.approx(
            float(np.linalg.norm(towards_patch)), abs=1e-6
        )


# %% what is wiped


def test_every_patch_is_covered_in_lanes_rather_than_in_one_line():
    """
    One lane is not a wipe: a patch is laid out in lanes side by side, so its path has
    to spread across them as well as run along them.
    """
    demonstration, world = populated_demonstration()
    sponge = min(demonstration.sponge_scale.x, demonstration.sponge_scale.y)

    for action in wiping_actions(demonstration, world):
        points = np.array([point.to_np().flatten()[:3] for point in action._waypoints])
        assert np.ptp(points[:, 0]) > sponge
        assert np.ptp(points[:, 1]) > sponge


def test_the_wipe_covers_the_border_of_the_table_the_robot_can_stand_at():
    """
    A parked HSRB reaches a strip along the edge it stands at and not the middle of a
    table this deep, so that border is what the placements together have to cover.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    footprint = table.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    sponge = min(demonstration.sponge_scale.x, demonstration.sponge_scale.y)

    waypoints = wiped_points(demonstration, world)

    assert footprint.max_x - waypoints[:, 0].max() < sponge
    assert waypoints[:, 1].min() - footprint.min_y < sponge
    assert np.ptp(waypoints[:, 1]) > 0.5 * footprint.width


def test_the_table_is_wiped_in_lanes():
    """
    A spiral and a shear each cover one patch in the middle of a surface; lanes are what
    covers a table.
    """
    assert WipeTableDemonstration(used_robot=HSRB).technique is WipingTechnique.SPREAD


def test_the_patch_stays_within_what_the_arm_reaches_flat():
    """
    The base is held while the tool presses, so every corner of the wiped patch has to
    be one the arm can hold the pad flat on from where the robot stands.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]

    for standing in demonstration.standing_poses(world, table):
        patch = demonstration.reachable_patch(world, standing)
        base = standing.to_np()[:2, 3]
        centre = patch.centre.to_np()
        heading = math.atan2(centre[1, 0], centre[0, 0])
        along = np.array([math.cos(heading), math.sin(heading)])
        across = np.array([-along[1], along[0]])
        for sideways in (-0.5, 0.5):
            for inwards in (-0.5, 0.5):
                corner = (
                    centre[:2, 3]
                    + along * sideways * patch.length
                    + across * inwards * patch.width
                )
                assert np.linalg.norm(corner - base) <= demonstration.flat_reach + 1e-9


def test_the_patches_cover_the_sides_the_robot_can_stand_at():
    """
    One stop covers only what the arm reaches from it, so the stops have to be close
    enough along an edge that no strip is left between their patches.
    """
    demonstration, world = populated_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]

    poses = demonstration.standing_poses(world, table)

    assert len(poses) > 1
    by_heading = {}
    for standing in poses:
        pose = standing.to_np()
        heading = round(math.atan2(pose[1, 0], pose[0, 0]), 6)
        by_heading.setdefault(heading, []).append(standing)
    for heading, stops in by_heading.items():
        along = np.array([-math.sin(heading), math.cos(heading)])
        offsets = sorted(float(stop.to_np()[:2, 3] @ along) for stop in stops)
        covered = demonstration.reachable_patch(world, stops[0]).length
        for lower, upper in zip(offsets, offsets[1:]):
            assert upper - lower <= covered + 1e-9


def test_every_wipe_fits_in_the_time_the_demonstration_allows_one_action():
    """
    A simulated action is given up on once it outlasts the budget the plan gives each of
    them, so the lanes of one patch have to be travelled inside it however many they
    take.
    """
    demonstration, world = populated_demonstration()
    budget = demonstration.simulation_time_per_action.total_seconds()

    for action in wiping_actions(demonstration, world):
        points = np.array(
            [point.to_np().flatten()[:3] for point in action._waypoints]
        )
        travel = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

        assert travel / demonstration.wipe_speed < budget


# %% the uneven table


def ridged_demonstration():
    demonstration = WipeWavyTableDemonstration(used_robot=HSRB)
    return demonstration, demonstration.build_simulated_world()


def ridge_boxes(world):
    return [
        body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()
        for body in world.bodies
        if body.name.name.startswith("ridge_")
    ]


def test_every_ridge_lies_over_the_table():
    """
    A ridge past the edge stands in mid air with nothing under it, and there is where
    the base drives.
    """
    demonstration, world = ridged_demonstration()
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    footprint = table.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()

    boxes = ridge_boxes(world)

    assert boxes
    for box in boxes:
        assert footprint.min_x <= box.min_x and box.max_x <= footprint.max_x
        assert footprint.min_y <= box.min_y and box.max_y <= footprint.max_y


def test_the_wipe_rides_the_ridges():
    """
    The ridges are what this demonstration measures, so the lanes have to cross them
    rather than run along a flat stretch between two of them.
    """
    demonstration, world = ridged_demonstration()
    waypoints = wiped_points(demonstration, world)

    boxes = ridge_boxes(world)

    centres = np.array(
        [
            [0.5 * (box.min_x + box.max_x), 0.5 * (box.min_y + box.max_y)]
            for box in boxes
        ]
    )
    crossed = (
        (centres[:, 0] >= waypoints[:, 0].min())
        & (centres[:, 0] <= waypoints[:, 0].max())
        & (centres[:, 1] >= waypoints[:, 1].min())
        & (centres[:, 1] <= waypoints[:, 1].max())
    )
    assert crossed.sum() > 1


def test_only_the_crest_of_each_ridge_stands_out_of_the_table():
    """
    The tool rides a wave rather than climbing a step, so the rest of each cylinder is
    sunk into the surface.
    """
    demonstration, world = ridged_demonstration()
    surface = demonstration.contact_model(world).surface_height

    for box in ridge_boxes(world):
        assert box.max_z == pytest.approx(
            surface + demonstration.wave_amplitude, abs=1e-6
        )


# %% force control


def test_the_real_robot_reads_its_own_sensor():
    """The simulated contact model stands in for the sensor, so a real run must not get
    one: the controller follows the sensor's own topic instead."""
    demonstration, world = populated_demonstration(
        executor_type=RealRobotPlanExecutor
    )

    assert demonstration.wrench_inputs(world).synchronizers == []


def test_a_simulated_run_supplies_its_own_contact():
    demonstration, world = populated_demonstration()

    inputs = demonstration.wrench_inputs(world)

    assert [type(reading) for reading in inputs.synchronizers] == [
        SimulatedContactWrench
    ]


def test_the_contact_reading_leaves_the_kinematic_state_alone():
    """
    A wrench is a measurement, not a degree of freedom.

    Reporting a write would recompute the forward kinematics once per reading.
    """
    demonstration, world = populated_demonstration()
    contact = demonstration.wrench_inputs(world).synchronizers[0]

    assert contact.apply() is False


def test_the_wipe_presses_into_the_surface_and_travels_across_it():
    """
    End to end: the admittance finds the table without being told its exact height, then
    holds the press while the tool covers ground.

    One band, because the press is what this exercises; covering the whole table is many
    times the travel and says nothing more about it.
    """
    demonstration, world = populated_demonstration(placements=1)
    table = world.get_semantic_annotations_by_type(DiningTable)[0]
    surface_height = demonstration.contact_model(world).surface_height
    start = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(demonstration.sponge_name)
    )[:3, 3].copy()

    world = demonstration.run()

    sponge = world.get_body_by_name(demonstration.sponge_name)
    end = world.compute_forward_kinematics_np(world.root, sponge)[:3, 3]
    assert np.linalg.norm(end[:2] - start[:2]) > 0.1
    # The press settles where the surface pushes back as hard as it is pressed. What
    # touches is the tool's own face, so that is what sinks into the surface, and its
    # frame sits half a tool above it.
    face_below_the_frame = demonstration.sponge_scale.z / 2
    np.testing.assert_allclose(
        end[2] - surface_height,
        face_below_the_frame
        - demonstration.press.force.to_np().item(2) / SimulatedContactWrench.stiffness,
        atol=5e-3,
    )


def test_the_surface_never_pushes_harder_than_it_says_it_does():
    """
    The cap bounds the push of the whole face, not of each spring under it, or a tool
    arriving fast reports many times the force the model documents.
    """
    demonstration, world = populated_demonstration()
    contact = demonstration.wrench_inputs(world).synchronizers[0]
    sponge = world.get_body_by_name(demonstration.sponge_name)
    sensor = ForceTorqueSensor.for_tip(world, sponge)
    # The tool's own body stands in for the surface, so every sampled point of its face
    # is over it wherever the robot happens to be.
    contact.surface = sponge
    contact.surface_height = (
        world.compute_forward_kinematics_np(world.root, sponge)[2, 3] + 1.0
    )

    force, _ = contact.contact_wrench(sensor)

    assert np.linalg.norm(force) <= contact.maximum_force + 1e-6


def test_the_demonstration_simulates_unless_it_is_told_otherwise():
    """
    The module documents a simulated default, and one that drives the real robot unless
    told otherwise is the wrong way round.
    """
    default = signature(main).parameters["executor_type"].default

    assert default is KinematicallySimulatedPlanExecutor
