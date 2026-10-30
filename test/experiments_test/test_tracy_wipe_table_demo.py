import numpy as np
import pytest

from ..pytest_environment import runs_in_continuous_integration

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.actions.composite.tool_based import WipingAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from coraplex.robot_plans.motions.gripper import LowerUntilContactMotion
from coraplex.plans.plan_node import MotionNode
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from experiments.tracy_wipe_table_demo.demo import TracyWipeTableDemonstration
from experiments.wipe_table_demo.demo import SimulatedContactWrench
from semantic_digital_twin.adapters.sensors.force_torque import (
    PhysicallySimulatedContactWrench,
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

    sponge = world.get_body_by_name(demonstration.sponge_name)

    assert world.get_semantic_annotations_by_type(Sponge)
    assert sponge.parent_connection.parent.name.name == "l_gripper_tool_frame"


def sponge_and_fingertips(demonstration, world):
    """
    :return: What the sponge and the fingertips holding it occupy, in the tool frame.
    """
    tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    fingertips = demonstration.wiping_arm(world).end_effector.fingertips
    sponge = world.get_body_by_name(demonstration.sponge_name)
    return (
        sponge.collision.as_bounding_box_collection_in_frame(tool_frame).bounding_box(),
        [
            tip.collision.as_bounding_box_collection_in_frame(tool_frame).bounding_box()
            for tip in fingertips.both
        ],
    )


def test_the_sponge_sticks_out_past_the_fingers_that_hold_it():
    """
    A sponge the fingers reach around cannot press on anything, and one hanging past
    them is held by nothing.
    """
    demonstration, world = populated_demonstration()

    sponge, tips = sponge_and_fingertips(demonstration, world)

    tips_end_at = max(tip.max_z for tip in tips)
    assert sponge.max_z == pytest.approx(tips_end_at + demonstration.sponge_protrusion)
    assert sponge.min_z < tips_end_at


def test_the_fingers_close_on_the_sponges_narrow_side():
    """
    The long side between the fingers leaves the sponge turned across its own sweep, and
    a gripper that cannot open that far cannot take it at all.
    """
    demonstration, world = populated_demonstration()

    sponge, _ = sponge_and_fingertips(demonstration, world)

    closing_axis = (
        demonstration.wiping_arm(world).end_effector.fingertips.closing_axis.to_np()
    ).flatten()
    width = sponge.max_x - sponge.min_x
    length = sponge.max_y - sponge.min_y
    np.testing.assert_allclose(np.abs(closing_axis[:3]), [1.0, 0.0, 0.0], atol=1e-9)
    assert width == pytest.approx(
        min(float(demonstration.sponge_scale.x), float(demonstration.sponge_scale.y))
    )
    assert length == pytest.approx(
        max(float(demonstration.sponge_scale.x), float(demonstration.sponge_scale.y))
    )


def test_the_sponge_is_measured_by_the_sensor_of_the_arm_that_holds_it():
    """
    The other arm's sensor feels nothing, so a wipe resolving it would never press.
    """
    demonstration, world = populated_demonstration()

    sensor = ForceTorqueSensor.for_tip(
        world, world.get_body_by_name(demonstration.sponge_name)
    )

    assert isinstance(sensor, TracyLeftForceTorqueSensor)


# %% the bench


def test_the_bench_top_is_the_surface_and_not_the_fixture_standing_on_it():
    demonstration, world = populated_demonstration()
    bench = world.get_body_by_name(demonstration.bench_name)
    boxes = list(bench.collision.as_bounding_box_collection_in_frame(world.root))
    spot = demonstration.touchdown_pose(world).to_np()[:3, 3]

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
    bench = world.get_body_by_name(demonstration.bench_name)
    footprint = bench.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    centre = demonstration.touchdown_pose(world).to_np()[:3, 3]

    assert footprint.min_x <= centre[0] - demonstration.patch_length / 2
    assert centre[0] + demonstration.patch_length / 2 <= footprint.max_x
    assert footprint.min_y <= centre[1] - demonstration.patch_width / 2
    assert centre[1] + demonstration.patch_width / 2 <= footprint.max_y


def test_the_wiped_patch_stays_clear_of_what_stands_on_the_bench():
    """
    The bench carries a camera pole whose lower brace lies flat on the top, and that pole
    is a body of its own: reading the bench's own shapes alone misses it. What sweeps the
    patch is the whole hand, which reaches almost twice as far sideways as the sponge and
    is free to spin about the surface normal, so the sponge alone clearing it is not
    enough either.
    """
    demonstration, world = populated_demonstration()
    top = demonstration.bench_height(world)
    end_effector = demonstration.wiping_arm(world).end_effector
    hand = [
        box
        for body in end_effector.bodies_with_collision
        for box in body.collision.as_bounding_box_collection_in_frame(
            end_effector.tool_frame
        )
    ]
    reach = max(
        max(abs(corner) for corner in (box.min_x, box.max_x, box.min_y, box.max_y))
        for box in hand
    )
    swept_height = max(box.max_z - box.min_z for box in hand)
    centre = demonstration.touchdown_pose(world).to_np()[:3, 3]
    reach_x = demonstration.patch_length / 2 + reach + demonstration.obstacle_clearance
    reach_y = demonstration.patch_width / 2 + reach + demonstration.obstacle_clearance
    patch_min_x, patch_max_x = centre[0] - reach_x, centre[0] + reach_x
    patch_min_y, patch_max_y = centre[1] - reach_y, centre[1] + reach_y

    standing_on_the_bench = [
        box
        for body in world.bodies_with_collision
        for box in body.collision.as_bounding_box_collection_in_frame(world.root)
        if top < box.max_z and box.min_z < top + swept_height
    ]

    assert standing_on_the_bench
    for box in standing_on_the_bench:
        assert (
            box.max_x <= patch_min_x
            or patch_max_x <= box.min_x
            or box.max_y <= patch_min_y
            or patch_max_y <= box.min_y
        ), f"the hand would come within {demonstration.obstacle_clearance} m of {box}"


# %% force control


def test_the_real_robot_reads_its_own_sensor():
    demonstration, world = populated_demonstration(execution_type=ExecutionType.REAL)

    assert demonstration.world_inputs(world) == []


def test_a_simulated_run_feels_the_bench_it_wipes():
    demonstration, world = populated_demonstration()

    inputs = demonstration.world_inputs(world)

    assert [type(reading) for reading in inputs] == [SimulatedContactWrench]
    assert inputs[0].surface is world.get_body_by_name(demonstration.bench_name)


def test_the_wipe_presses_into_the_bench_and_travels_across_it():
    """
    End to end: the admittance finds the bench without being told its exact height, then
    holds the press while the tool covers ground.
    """
    demonstration, world = populated_demonstration()
    bench_height = demonstration.bench_height(world)
    start = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(demonstration.sponge_name)
    )[:3, 3].copy()

    world = demonstration.run()

    end = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name(demonstration.sponge_name)
    )[:3, 3]
    assert np.linalg.norm(end[:2] - start[:2]) > 0.1
    # The press settles where the surface pushes back as hard as it is pressed. What
    # touches is the tool's own face, so that is what sinks into the surface, and its
    # frame sits half a tool above it.
    face_below_the_frame = float(demonstration.sponge_scale.z) / 2
    np.testing.assert_allclose(
        end[2] - bench_height,
        face_below_the_frame
        - demonstration.press.force.to_np().item(2) / SimulatedContactWrench.stiffness,
        atol=5e-3,
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


def approach_steps(steps):
    """
    :return: The steps that bring the tool over the patch, in the order they run.
    """
    return [step for step in steps if isinstance(step, MoveManipulatorAction)]


def test_the_tool_is_lowered_onto_the_bench_before_it_is_wiped():
    """
    Where the surface really is is measured, not assumed: the wipe starts from contact.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)

    assert steps.index(approach_steps(steps)[-1]) < index_of(
        steps, LowerUntilContactMotion
    )
    assert index_of(steps, LowerUntilContactMotion) < index_of(steps, WipingAction)


def test_the_descent_aims_past_the_bench_so_only_contact_ends_it():
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    descent = steps[index_of(steps, LowerUntilContactMotion)]

    goal_height = descent.goal_point.to_np().flatten()[2]
    assert goal_height < demonstration.bench_height(world)


def test_the_descent_is_capped_to_the_speed_it_is_asked_to_lower_at():
    """
    A descent's own ``descent_velocity`` is a reference velocity, which normalises the
    goal's error and does not limit anything, so without a cap the tool arrives at
    whatever speed the arm can manage and the pad meets the bench at ten times the
    force.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    descent = steps[index_of(steps, LowerUntilContactMotion)]

    assert descent.max_linear_velocity == demonstration.descent_velocity
    assert descent.descent_velocity == demonstration.descent_velocity


def test_the_wipe_is_capped_to_a_travel_speed():
    """
    Capping the speed is what lets the tracked point run far enough ahead to pull the
    tool along, instead of the pull being weakened to slow the tool down.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)
    wipe = steps[index_of(steps, WipingAction)]

    assert wipe.max_linear_velocity == demonstration.wipe_speed


def test_the_tool_points_at_the_bench_while_it_is_lowered():
    """
    Pointing away would put the hand under the tool and drive it into the bench, and the
    wipe's own alignment cannot flip the tool through the pose where it starts inverted.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)

    for approach in approach_steps(steps):
        tool_z_in_world = approach.target_pose.to_np()[:3, 2]
        np.testing.assert_allclose(tool_z_in_world, [0.0, 0.0, -1.0], atol=1e-9)


def test_the_tool_turns_onto_the_bench_well_above_it_and_only_then_comes_down():
    """
    A tool still turning on its way down meets the bench edge with the hand rather than
    with the sponge.
    """
    demonstration, world = populated_demonstration()

    steps = plan_steps(demonstration, world)

    heights = [
        approach.target_pose.to_np()[2, 3] - demonstration.bench_height(world)
        for approach in approach_steps(steps)
    ]
    assert heights == pytest.approx(
        [demonstration.turning_height, demonstration.approach_height]
    )


# %% the charts the controller compiles


def motions_of(plan_step) -> list:
    """
    :param plan_step: An action or a motion of the plan.
    :return: The motions it expands into, which are what carry a chart.
    """
    if hasattr(plan_step, "_motion_chart"):
        return [plan_step]
    plan_step.expand()
    return [
        node.designator
        for node in plan_step.plan.all_nodes
        if isinstance(node, MotionNode) and hasattr(node.designator, "_motion_chart")
    ]


def test_every_motion_of_the_plan_compiles():
    """
    Building a chart is not enough: the controller compiles it before running it, and
    wiring a node to one it does not share a chart with is only rejected there.
    """
    demonstration, world = populated_demonstration()

    compiled = []
    for step in plan_steps(demonstration, world):
        for motion in motions_of(step):
            chart = MotionStatechart()
            chart.add_node(motion.motion_chart)
            Executor(MotionStatechartContext(world=world)).compile(
                motion_statechart=chart
            )
            compiled.append(type(motion).__name__)

    assert "MoveTCPWaypointsAlignedMotion" in compiled
    assert "LowerUntilContactMotion" in compiled


# %% the physics a run is performed in


def test_a_kinematically_simulated_run_needs_no_physics():
    """
    Nothing resolves contacts in a kinematic run, so its press is modelled instead.
    """
    demonstration, world = populated_demonstration()

    assert demonstration.start_physics(world) is None
    assert [type(reading) for reading in demonstration.world_inputs(world)] == [
        SimulatedContactWrench
    ]


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_physically_simulated_run_feels_what_the_physics_resolves():
    """
    The press is only as good as the contact it is measured against, so a run that
    simulates the physics reads its wrench from there rather than from a model of the
    bench.
    """
    demonstration, world = populated_demonstration(
        execution_type=ExecutionType.PHYSICALLY_SIMULATED
    )

    demonstration.simulation = demonstration.start_physics(world)
    try:
        inputs = demonstration.world_inputs(world)
        assert [type(reading) for reading in inputs] == [
            PhysicallySimulatedContactWrench
        ]
        assert inputs[0].simulation is demonstration.simulation
    finally:
        demonstration.tear_down()
