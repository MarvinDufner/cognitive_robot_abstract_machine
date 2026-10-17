import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.force_monitors import ContactForceReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world import World

TOOL_FRAME = "hand_gripper_tool_frame"
THRESHOLD = 2.0


# %% helpers


def build_monitor(world, **arguments):
    """
    Compile a chart holding one contact monitor and return it with its context.
    """
    chart = MotionStatechart()
    chart.add_node(
        monitor := ContactForceReached(
            name="contact",
            tip_link=world.get_body_by_name(TOOL_FRAME),
            threshold=THRESHOLD,
            **arguments,
        )
    )
    chart.add_node(EndMotion.when_true(monitor))
    context = MotionStatechartContext(world=world)
    Executor(context).compile(motion_statechart=chart)
    return monitor, context


def sensor_of(world) -> ForceTorqueSensor:
    return ForceTorqueSensor.for_tip(world, world.get_body_by_name(TOOL_FRAME))


def push(
    monitor: ContactForceReached,
    context: MotionStatechartContext,
    world: World,
    force: np.ndarray,
    readings: int,
) -> ObservationStateValues:
    """
    Feed the same force through the monitor a number of times.

    :return: What it observed on the last of them.
    """
    for _ in range(readings):
        sensor_of(world).write_wrench(force, np.zeros(3))
        observation = monitor.on_tick(context)
    return observation


# %% observing contact


def test_an_unread_sensor_is_not_yet_out_of_contact(hsr_world_copy):
    """
    Before the first reading the wrench is unknown, not zero, so reporting no contact
    would let a descent that never started count as finished.
    """
    monitor, context = build_monitor(hsr_world_copy)

    assert monitor.on_tick(context) is ObservationStateValues.UNKNOWN


def test_a_sensor_measuring_nothing_reports_no_contact(hsr_world_copy):
    monitor, context = build_monitor(hsr_world_copy)
    sensor_of(hsr_world_copy).write_wrench(np.zeros(3), np.zeros(3))

    assert monitor.on_tick(context) is ObservationStateValues.FALSE


@pytest.mark.parametrize("axis", range(3))
def test_contact_is_reported_whichever_way_the_surface_pushes(hsr_world_copy, axis):
    """
    The tool meets a surface at whatever angle it is held, so only the magnitude counts.
    """
    monitor, context = build_monitor(hsr_world_copy)
    force = np.zeros(3)
    force[axis] = -2.0 * THRESHOLD

    observed = push(
        monitor, context, hsr_world_copy, force, monitor.confirming_readings
    )

    assert observed is ObservationStateValues.TRUE


def test_a_touch_below_the_threshold_is_not_contact_yet(hsr_world_copy):
    """
    The threshold is what separates contact from the noise of carrying the tool.
    """
    monitor, context = build_monitor(hsr_world_copy)

    sensor_of(hsr_world_copy).write_wrench(
        np.array([0.0, 0.0, 0.5 * THRESHOLD]), np.zeros(3)
    )

    assert monitor.on_tick(context) is ObservationStateValues.FALSE


def test_the_torque_of_a_carried_tool_is_not_mistaken_for_contact(hsr_world_copy):
    """
    A tool hanging off the sensor loads it in torque without touching anything.
    """
    monitor, context = build_monitor(hsr_world_copy)

    sensor_of(hsr_world_copy).write_wrench(np.zeros(3), np.ones(3) * 10.0)

    assert monitor.on_tick(context) is ObservationStateValues.FALSE


# %% contact, unlike a spike, stays


def test_one_reading_above_the_threshold_is_not_contact(hsr_world_copy):
    """
    A sensor's data path produces isolated spikes many times the force a tool touches
    down with, and acting on one ends a descent in mid-air.
    """
    monitor, context = build_monitor(hsr_world_copy)

    observed = push(
        monitor, context, hsr_world_copy, np.array([0.0, 0.0, -50.0 * THRESHOLD]), 1
    )

    assert monitor.confirming_readings > 1
    assert observed is ObservationStateValues.FALSE


def test_contact_is_reported_once_the_readings_keep_coming(hsr_world_copy):
    monitor, context = build_monitor(hsr_world_copy)
    touching = np.array([0.0, 0.0, -2.0 * THRESHOLD])

    before_the_last = push(
        monitor, context, hsr_world_copy, touching, monitor.confirming_readings - 1
    )
    confirming = push(monitor, context, hsr_world_copy, touching, 1)

    assert before_the_last is ObservationStateValues.FALSE
    assert confirming is ObservationStateValues.TRUE


def test_a_reading_of_nothing_in_between_starts_the_count_over(hsr_world_copy):
    """
    Spikes are isolated, so what separates them from contact is that the readings in
    between are back below the threshold.
    """
    monitor, context = build_monitor(hsr_world_copy)
    touching = np.array([0.0, 0.0, -2.0 * THRESHOLD])

    for _ in range(monitor.confirming_readings):
        push(monitor, context, hsr_world_copy, touching, 1)
        observed = push(monitor, context, hsr_world_copy, np.zeros(3), 1)

    assert observed is ObservationStateValues.FALSE
    assert monitor._readings_above_threshold == 0


def test_a_monitor_asked_for_one_reading_reports_it_at_once(hsr_world_copy):
    """
    A sensor whose readings are already filtered needs no confirmation, and waiting for
    one would only add the delay back.
    """
    monitor, context = build_monitor(hsr_world_copy, confirming_readings=1)

    observed = push(
        monitor, context, hsr_world_copy, np.array([0.0, 0.0, -2.0 * THRESHOLD]), 1
    )

    assert observed is ObservationStateValues.TRUE


def test_a_reset_monitor_waits_for_the_whole_count_again(hsr_world_copy):
    """
    The node is reset to be run again, and a count left standing would let the next
    descent finish on its first spike.
    """
    monitor, context = build_monitor(hsr_world_copy)
    touching = np.array([0.0, 0.0, -2.0 * THRESHOLD])
    push(monitor, context, hsr_world_copy, touching, monitor.confirming_readings - 1)

    monitor.on_reset(context)
    observed = push(monitor, context, hsr_world_copy, touching, 1)

    assert observed is ObservationStateValues.FALSE
