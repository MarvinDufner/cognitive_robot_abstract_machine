import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import NonPositiveVirtualMassError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.admittance_tasks import (
    AdmittanceCartesianTrajectory,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.spatial_types import Point3, Vector3

TOOL_FRAME = "hand_gripper_tool_frame"
SETTLING_TICKS = 400


# %% helpers


def stroke(world, waypoints=5):
    return [
        Point3(x=0.45 + 0.03 * index, y=0.0, z=0.60, reference_frame=world.root)
        for index in range(waypoints)
    ]


def build_admittance(world, waypoints=5, **task_arguments):
    """
    Compile a chart holding one admittance trajectory and return it with its context.
    """
    chart = MotionStatechart()
    chart.add_node(
        task := AdmittanceCartesianTrajectory(
            name="admittance",
            root_link=world.root,
            tip_link=world.get_body_by_name(TOOL_FRAME),
            goal_points=stroke(world, waypoints),
            maximum_skip_ahead=2,
            **task_arguments,
        )
    )
    chart.add_node(EndMotion.when_true(task))
    context = MotionStatechartContext(world=world)
    Executor(context).compile(motion_statechart=chart)
    return task, context, chart


def press(task, context, sensor, force, ticks=SETTLING_TICKS):
    """
    Feed a constant force through the task and return the offset history.
    """
    offsets = []
    for _ in range(ticks):
        sensor.write_wrench(force, np.zeros(3))
        task.on_tick(context)
        offsets.append(task._state[:3].copy())
    return np.array(offsets)


def sensor_of(world):
    return ForceTorqueSensor.for_tip(world, world.get_body_by_name(TOOL_FRAME))


# %% one task for the whole trajectory


def test_a_whole_stroke_is_one_task(hsr_world_copy):
    """
    One task per waypoint would restart the compliance at every waypoint.
    """
    _, _, chart = build_admittance(hsr_world_copy, waypoints=12)

    assert (
        len([n for n in chart.nodes if isinstance(n, AdmittanceCartesianTrajectory)])
        == 1
    )


def test_the_offset_survives_moving_to_the_next_waypoint(hsr_world_copy):
    """
    The compliance offset describes where the surface pushed the tool to, which does not
    change just because the tool has moved along the stroke.
    """
    task, context, _ = build_admittance(hsr_world_copy, waypoints=5)
    sensor = sensor_of(hsr_world_copy)

    press(task, context, sensor, np.array([0.0, 0.0, -10.0]), ticks=40)
    offset_before = task._state[:3].copy()
    assert np.any(offset_before != 0.0)

    # Advance along the stroke, as tracking the trajectory does.
    task.current_index += 1
    sensor.write_wrench(np.array([0.0, 0.0, -10.0]), np.zeros(3))
    task.on_tick(context)

    assert np.linalg.norm(task._state[:3] - offset_before) < 0.01


# %% the dynamics


def test_the_compiled_step_matches_the_dynamics_it_stands_for(hsr_world_copy):
    mass = Vector3(x=1.5, y=2.0, z=0.8)
    damping = Vector3(x=20.0, y=35.0, z=100.0)
    stiffness = Vector3(x=0.0, y=5.0, z=2000.0)
    desired_force = Vector3(x=0.0, y=1.0, z=8.0)
    task, context, _ = build_admittance(
        hsr_world_copy,
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        desired_force=desired_force,
    )
    sensor = sensor_of(hsr_world_copy)
    control_dt = context.qp_controller_config.control_dt
    m = mass.to_np()[:3].flatten()
    d = damping.to_np()[:3].flatten()
    k = stiffness.to_np()[:3].flatten()
    f_desired = desired_force.to_np()[:3].flatten()
    generator = np.random.default_rng(0)

    for _ in range(50):
        state = generator.normal(scale=0.05, size=6)
        wrench = generator.normal(scale=10.0, size=6)
        sensor.wrench_data.data[:] = wrench

        stepped = np.array(
            task._compiled_step(
                hsr_world_copy.state.positions, sensor.wrench_data.data, state
            )
        ).flatten()

        goal_R_sensor = hsr_world_copy.compute_forward_kinematics_np(
            hsr_world_copy.root, sensor.root
        )[:3, :3]
        force_error = goal_R_sensor @ wrench[:3] - f_desired
        velocity = (m * state[3:] + (force_error - k * state[:3]) * control_dt) / (
            m + d * control_dt + k * control_dt**2
        )
        np.testing.assert_allclose(
            stepped,
            np.concatenate([state[:3] + velocity * control_dt, velocity]),
            atol=1e-12,
        )


def test_the_step_only_depends_on_fixed_size_inputs(hsr_world_copy):
    """
    Its parameter groups must not grow with the chart, or a longer stroke would compile
    a larger problem.
    """
    task, _, _ = build_admittance(hsr_world_copy, waypoints=40)

    assert len(sensor_of(hsr_world_copy).wrench_data.variables) == 6
    assert task._state.shape == (6,)


def test_a_mass_that_cannot_be_integrated_is_rejected(hsr_world_copy):
    with pytest.raises(NonPositiveVirtualMassError):
        build_admittance(hsr_world_copy, mass=Vector3(x=1.0, y=1.0, z=0.0))


# %% stability and contact seeking


def test_pure_damping_settles_at_the_terminal_velocity(hsr_world_copy):
    damping = 20.0
    pressing_force = np.array([0.0, 0.0, -10.0])
    task, context, _ = build_admittance(
        hsr_world_copy,
        mass=Vector3(x=1.0, y=1.0, z=1.0),
        damping=Vector3(x=damping, y=damping, z=damping),
    )
    sensor = sensor_of(hsr_world_copy)

    press(task, context, sensor, pressing_force)

    goal_R_sensor = hsr_world_copy.compute_forward_kinematics_np(
        hsr_world_copy.root, sensor.root
    )[:3, :3]
    np.testing.assert_allclose(
        task._state[3:], goal_R_sensor @ pressing_force / damping, atol=1e-6
    )


def test_a_stiff_axis_settles_without_bouncing(hsr_world_copy):
    stiffness = 2000.0
    task, context, _ = build_admittance(
        hsr_world_copy,
        mass=Vector3(x=1.0, y=1.0, z=1.0),
        damping=Vector3(x=100.0, y=100.0, z=100.0),
        stiffness=Vector3(x=stiffness, y=stiffness, z=stiffness),
        desired_force=Vector3(z=8.0),
    )
    sensor = sensor_of(hsr_world_copy)

    offsets = press(task, context, sensor, np.zeros(3))

    distances = np.linalg.norm(offsets - offsets[-1], axis=1)
    assert np.all(np.diff(distances) <= 1e-9)


def test_the_tool_seeks_the_surface_when_it_feels_nothing(hsr_world_copy):
    """
    Out of contact the whole desired force is an error, so the offset moves the tool
    towards the surface until it touches.

    That is why the commanded height need not be exact.
    """
    task, context, _ = build_admittance(hsr_world_copy, desired_force=Vector3(z=8.0))
    sensor = sensor_of(hsr_world_copy)

    offsets = press(task, context, sensor, np.zeros(3), ticks=20)

    assert offsets[-1][2] < offsets[0][2]


def test_the_offset_is_forgotten_on_reset(hsr_world_copy):
    task, context, _ = build_admittance(hsr_world_copy)
    sensor = sensor_of(hsr_world_copy)
    press(task, context, sensor, np.array([0.0, 0.0, -10.0]), ticks=20)
    assert np.any(task._state != 0.0)

    task.on_reset(context)

    np.testing.assert_array_equal(task._state, np.zeros(6))


def test_an_unread_sensor_leaves_the_offset_alone(hsr_world_copy):
    """
    Before the first reading the wrench is not zero, it is unknown, so integrating it
    would invent a contact.
    """
    task, context, _ = build_admittance(hsr_world_copy)

    task.on_tick(context)

    np.testing.assert_array_equal(task._state, np.zeros(6))
