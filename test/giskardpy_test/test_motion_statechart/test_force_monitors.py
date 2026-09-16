import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.force_monitors import ContactForceReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor

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

    sensor_of(hsr_world_copy).write_wrench(force, np.zeros(3))

    assert monitor.on_tick(context) is ObservationStateValues.TRUE


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
