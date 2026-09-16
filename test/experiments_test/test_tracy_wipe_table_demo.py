import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from coraplex.robot_plans.motions.gripper import LowerUntilContactMotion
from experiments.tracy_wipe_table_demo.demo import (
    APPROACH_HEIGHT,
    BENCH_NAME,
    OBSTACLE_CLEARANCE,
    UNMODELLED_ADAPTER_LENGTH,
    WIPED_PATCH_LENGTH,
    WIPED_PATCH_WIDTH,
    TracyWipeTableDemonstration,
)
from experiments.wipe_table_demo.demo import (
    CONTACT_STIFFNESS,
    PRESS_FORCE,
    SPONGE_NAME,
    SPONGE_SCALE,
    SimulatedContactWrench,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.robots.tracy import Tracy, TracyLeftForceTorqueSensor
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge


def populated_demonstration(**arguments):
    demonstration = TracyWipeTableDemonstration(used_robot=Tracy, **arguments)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return demonstration, world


# %% scene


def test_the_scene_counts_as_populated_once_the_sponge_is_attached():
    demonstration = TracyWipeTableDemonstration(used_robot=Tracy)
    world = demonstration.build_simulated_world()

    assert not demonstration.is_scene_populated(world)
    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_the_sponge_hangs_off_the_wiping_hand():
    demonstration, world = populated_demonstration()

    sponge = world.get_body_by_name(SPONGE_NAME)

    assert world.get_semantic_annotations_by_type(Sponge)
    assert sponge.parent_connection.parent.name.name == "l_gripper_tool_frame"


def test_the_sponge_hangs_where_the_hardware_puts_it_not_where_the_description_does():
    """
    The description is missing an adapter, so a sponge placed by it alone would sit that
    much too close to the hand, and every press would drive the tool that much through
    the surface.
    """
    demonstration, world = populated_demonstration()
    tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    sponge = world.get_body_by_name(SPONGE_NAME)

    along_the_tool = world.compute_forward_kinematics_np(tool_frame, sponge)[2, 3]

    assert along_the_tool == pytest.approx(
        UNMODELLED_ADAPTER_LENGTH + float(SPONGE_SCALE.z) / 2
    )


def test_the_sponge_is_measured_by_the_sensor_of_the_arm_that_holds_it():
    """
    The other arm's sensor feels nothing, so a wipe resolving it would never press.
    """
    demonstration, world = populated_demonstration()

    sensor = ForceTorqueSensor.for_tip(world, world.get_body_by_name(SPONGE_NAME))

    assert isinstance(sensor, TracyLeftForceTorqueSensor)


# %% the bench


def test_the_bench_top_is_the_surface_and_not_the_fixture_standing_on_it():
    demonstration, world = populated_demonstration()
    bench = world.get_body_by_name(BENCH_NAME)
    boxes = list(bench.collision.as_bounding_box_collection_in_frame(world.root))
    spot = demonstration.wiped_pose(world).to_np()[:3, 3]

    height = demonstration.bench_height(world)

    assert height < max(box.max_z for box in boxes)
    assert height == pytest.approx(
        max(
            box.max_z
            for box in boxes
            if box.min_x <= spot[0] <= box.max_x and box.min_y <= spot[1] <= box.max_y
        )
    )


def test_the_wiped_patch_lies_on_the_bench_top():
    demonstration, world = populated_demonstration()
    bench = world.get_body_by_name(BENCH_NAME)
    footprint = bench.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    centre = demonstration.wiped_pose(world).to_np()[:3, 3]

    assert footprint.min_x <= centre[0] - WIPED_PATCH_LENGTH / 2
    assert centre[0] + WIPED_PATCH_LENGTH / 2 <= footprint.max_x
    assert footprint.min_y <= centre[1] - WIPED_PATCH_WIDTH / 2
    assert centre[1] + WIPED_PATCH_WIDTH / 2 <= footprint.max_y


def test_the_wiped_patch_stays_clear_of_what_stands_on_the_bench():
    """
    The bench carries a camera pole whose lower brace lies flat on the top, and that pole
    is a body of its own: reading the bench's own shapes alone misses it.
    """
    demonstration, world = populated_demonstration()
    top = demonstration.bench_height(world)
    sponge = world.get_body_by_name(SPONGE_NAME)
    swept = sponge.collision.as_bounding_box_collection_in_frame(sponge).bounding_box()
    centre = demonstration.wiped_pose(world).to_np()[:3, 3]
    # What the sponge sweeps: the patch, widened by the sponge it is wiped with and by
    # the gap the patch is meant to keep.
    reach_x = (WIPED_PATCH_LENGTH + swept.max_x - swept.min_x) / 2 + OBSTACLE_CLEARANCE
    reach_y = (WIPED_PATCH_WIDTH + swept.max_y - swept.min_y) / 2 + OBSTACLE_CLEARANCE
    patch_min_x, patch_max_x = centre[0] - reach_x, centre[0] + reach_x
    patch_min_y, patch_max_y = centre[1] - reach_y, centre[1] + reach_y

    standing_on_the_bench = [
        box
        for body in world.bodies_with_collision
        for box in body.collision.as_bounding_box_collection_in_frame(world.root)
        if top < box.max_z and box.min_z < top + (swept.max_z - swept.min_z)
    ]

    assert standing_on_the_bench
    for box in standing_on_the_bench:
        assert (
            box.max_x <= patch_min_x
            or patch_max_x <= box.min_x
            or box.max_y <= patch_min_y
            or patch_max_y <= box.min_y
        ), f"the sponge would come within {OBSTACLE_CLEARANCE} m of {box}"


# %% force control


def test_the_real_robot_reads_its_own_sensor():
    demonstration, world = populated_demonstration(execution_type=ExecutionType.REAL)

    assert demonstration.world_inputs(world) == []


def test_a_simulated_run_feels_the_bench_it_wipes():
    demonstration, world = populated_demonstration()

    inputs = demonstration.world_inputs(world)

    assert [type(reading) for reading in inputs] == [SimulatedContactWrench]
    assert inputs[0].surface is world.get_body_by_name(BENCH_NAME)


def test_the_wipe_presses_into_the_bench_and_travels_across_it():
    """
    End to end: the admittance finds the bench without being told its exact height, then
    holds the press while the tool covers ground.
    """
    demonstration, world = populated_demonstration()
    bench_height = demonstration.bench_height(world)
    start = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(SPONGE_NAME)
    )[:3, 3].copy()

    world = demonstration.run()

    end = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(SPONGE_NAME)
    )[:3, 3]
    assert np.linalg.norm(end[:2] - start[:2]) > 0.1
    # The press settles where the surface pushes back as hard as it is pressed.
    np.testing.assert_allclose(
        end[2] - bench_height, -PRESS_FORCE / CONTACT_STIFFNESS, atol=2e-3
    )


# %% feeling for the bench


def plan_steps(demonstration, world):
    """
    :return: The plan's steps in the order they run.
    """
    plan = demonstration.build_plan(demonstration.build_context(world))
    return [
        node.designator if hasattr(node, "designator") else node
        for node in plan.children
    ]


def index_of(steps, step_type) -> int:
    """
    :return: Where the one step of that type sits in the plan.
    """
    found = [index for index, step in enumerate(steps) if isinstance(step, step_type)]
    assert len(found) == 1, f"expected exactly one {step_type.__name__}, got {found}"
    return found[0]


def test_the_tool_is_lowered_onto_the_bench_before_it_is_wiped():
    """
    Where the surface really is is measured, not assumed: the wipe starts from contact.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)

    assert index_of(steps, MoveManipulatorAction) < index_of(
        steps, LowerUntilContactMotion
    )
    assert index_of(steps, LowerUntilContactMotion) < index_of(steps, WipingAction)


def test_the_descent_aims_past_the_bench_so_only_contact_ends_it():
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    descent = steps[index_of(steps, LowerUntilContactMotion)]

    goal_height = descent.goal_point.to_np().flatten()[2]
    assert goal_height < demonstration.bench_height(world)


def test_the_tool_points_at_the_bench_while_it_is_lowered():
    """
    Pointing away would put the hand under the tool and drive it into the bench, and the
    wipe's own alignment cannot flip the tool through the pose where it starts inverted.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    approach = steps[index_of(steps, MoveManipulatorAction)]

    tool_z_in_world = approach.target_pose.to_np()[:3, 2]
    np.testing.assert_allclose(tool_z_in_world, [0.0, 0.0, -1.0], atol=1e-9)


def test_the_approach_waits_above_the_bench_rather_than_in_it():
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    approach = steps[index_of(steps, MoveManipulatorAction)]

    height_above_the_bench = approach.target_pose.to_np()[
        2, 3
    ] - demonstration.bench_height(world)
    assert height_above_the_bench == pytest.approx(APPROACH_HEIGHT)
