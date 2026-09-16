from dataclasses import dataclass

import numpy as np
import pytest
from std_srvs.srv import Trigger
from typing_extensions import Any, Type

from giskardpy.middleware.ros2.exceptions import (
    ForceTorqueSensorNotTared,
    WrenchCompensationUnavailable,
)
from giskardpy.ros2_tools.wrench_compensation_node import (
    GRAVITY,
    BiasEstimator,
    RetareOutcome,
    WrenchCompensationClient,
    WrenchGravityCompensator,
)
from semantic_digital_twin.robots.robot_parts import (
    ForceTorqueSensor,
    ForceTorqueSensorLoad,
)

UPRIGHT = np.eye(3)
"""
Sensor axes aligned with the gravity frame, so gravity points along sensor -z.
"""


def rotated_about_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def reading_at_rest(load: ForceTorqueSensorLoad, base_R_sensor: np.ndarray):
    """
    The raw wrench the identified model predicts when nothing touches the tool.
    """
    gravity_in_sensor = base_R_sensor.T @ np.array([0.0, 0.0, -GRAVITY])
    return (
        load.force_offset + load.mass * gravity_in_sensor,
        load.torque_offset + np.cross(load.first_moment, gravity_in_sensor),
    )


@pytest.fixture
def wrist_load(hsr_world_copy) -> ForceTorqueSensorLoad:
    """
    The identified wrist load, read off the robot annotation as the bringup does.
    """
    return ForceTorqueSensor.for_tip(
        hsr_world_copy, hsr_world_copy.get_body_by_name("hand_gripper_tool_frame")
    ).load


# %% removing the load


@pytest.mark.parametrize("angle", [0.0, 0.5, np.pi / 2, np.pi, -1.3])
def test_a_resting_sensor_reports_no_contact_in_any_pose(angle, wrist_load):
    """
    The load's weight swings through the sensor axes as the wrist turns, so it has to be
    removed using the live orientation rather than a fixed offset.
    """
    load = wrist_load
    compensator = WrenchGravityCompensator(load=load)
    base_R_sensor = rotated_about_x(angle)
    force_raw, torque_raw = reading_at_rest(load, base_R_sensor)

    force, torque = compensator.contact_wrench(force_raw, torque_raw, base_R_sensor)

    np.testing.assert_allclose(force, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-9)


def test_a_push_survives_the_compensation(wrist_load):
    load = wrist_load
    compensator = WrenchGravityCompensator(load=load)
    force_raw, torque_raw = reading_at_rest(load, UPRIGHT)
    push = np.array([0.0, 0.0, -8.0])

    force, _ = compensator.contact_wrench(force_raw + push, torque_raw, UPRIGHT)

    np.testing.assert_allclose(force, push, atol=1e-9)


def test_an_unidentified_sensor_passes_its_reading_through():
    """
    The default load is nothing mounted, so compensation must not invent a correction.
    """
    compensator = WrenchGravityCompensator()
    reading = np.array([1.0, -2.0, 3.0])

    force, torque = compensator.contact_wrench(reading, reading, UPRIGHT)

    np.testing.assert_allclose(force, reading)
    np.testing.assert_allclose(torque, reading)


def test_gravity_points_down_in_a_level_sensor():
    compensator = WrenchGravityCompensator()

    np.testing.assert_allclose(
        compensator.gravity_in_sensor(UPRIGHT), [0.0, 0.0, -GRAVITY]
    )


# %% re-taring


def test_a_still_sensor_yields_its_mean_residual():
    estimator = BiasEstimator(required_samples=10)
    residual = np.array([0.4, -0.2, 1.1])
    generator = np.random.default_rng(0)

    for _ in range(estimator.required_samples):
        estimator.add(residual + generator.normal(scale=0.01, size=3), residual)

    bias_force, bias_torque = estimator.result()
    np.testing.assert_allclose(bias_force, residual, atol=0.05)
    np.testing.assert_allclose(bias_torque, residual)


def test_a_moving_sensor_is_refused():
    """
    Averaging while the arm moves would fold real motion into the bias.
    """
    estimator = BiasEstimator(required_samples=10, stationary_force_spread=0.5)

    for sample in range(estimator.required_samples):
        estimator.add(np.array([float(sample), 0.0, 0.0]), np.zeros(3))

    assert estimator.result() is None


def test_a_window_counts_what_it_collected():
    """
    A re-tare that ran out of time reports how far it got, which is what tells a reader
    the readings are arriving too slowly.
    """
    estimator = BiasEstimator(required_samples=4)

    assert estimator.collected == 0
    estimator.add(np.zeros(3), np.zeros(3))
    estimator.add(np.zeros(3), np.zeros(3))

    assert estimator.collected == 2

    estimator.reset()

    assert estimator.collected == 0


def test_a_window_reports_how_far_its_forces_spread():
    """
    A rejected window is only actionable if it says by how much, since the bar has to
    sit above whatever the sensor reads at rest.
    """
    estimator = BiasEstimator(required_samples=2)

    estimator.add(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    estimator.add(np.array([0.0, 0.0, 3.0]), np.zeros(3))

    assert estimator.force_spread == pytest.approx(1.0)


def test_noise_averages_out_between_the_halves_of_a_window():
    """
    A reading that only jitters has to look different from one that is walking away, or
    a re-tare cannot say which stopped it.
    """
    estimator = BiasEstimator(required_samples=4)
    for force in (-1.0, 1.0, -1.0, 1.0):
        estimator.add(np.array([0.0, 0.0, force]), np.zeros(3))

    assert estimator.force_shift == pytest.approx(0.0)


def test_a_drifting_reading_shows_up_as_a_shift_across_the_window():
    estimator = BiasEstimator(required_samples=4)
    for force in (0.0, 1.0, 2.0, 3.0):
        estimator.add(np.array([0.0, 0.0, force]), np.zeros(3))

    # The halves average 0.5 and 2.5.
    assert estimator.force_shift == pytest.approx(2.0)


def test_a_short_window_is_not_a_result():
    estimator = BiasEstimator(required_samples=10)

    assert not estimator.add(np.zeros(3), np.zeros(3))
    assert estimator.result() is None


def test_the_window_is_full_only_at_the_requested_count():
    estimator = BiasEstimator(required_samples=3)

    assert [estimator.add(np.zeros(3), np.zeros(3)) for _ in range(3)] == [
        False,
        False,
        True,
    ]


def test_a_measured_bias_is_subtracted_from_later_readings():
    compensator = WrenchGravityCompensator()
    drift = np.array([0.0, 0.0, 2.5])

    compensator.bias_force = drift

    force, _ = compensator.contact_wrench(drift, np.zeros(3), UPRIGHT)
    np.testing.assert_allclose(force, np.zeros(3))


# %% the load comes from the robot annotation


def test_the_wrist_load_is_read_from_the_robot(hsr_world_copy):
    sensor = ForceTorqueSensor.for_tip(
        hsr_world_copy, hsr_world_copy.get_body_by_name("hand_gripper_tool_frame")
    )

    assert sensor.load.mass > 0.0
    assert sensor.load.first_moment.shape == (3,)


# %% asking for a re-tare

RETARE_SERVICE = "/left_arm/wrench_compensation/retare"


@dataclass
class ServiceAnsweringOneOutcome:
    """
    Stands in for a service client, answering every request the same way.
    """

    outcome: RetareOutcome
    """
    What the answer reports.
    """

    offered: bool = True
    """
    Whether anything offers the service at all.
    """

    calls: int = 0
    """
    How many requests were answered.
    """

    def wait_for_service(self, timeout_sec: float) -> bool:
        return self.offered

    def call(self, request: Trigger.Request) -> Trigger.Response:
        self.calls += 1
        response = Trigger.Response()
        response.success = self.outcome is RetareOutcome.APPLIED
        response.message = self.outcome
        return response


@dataclass
class NodeOfferingOneService:
    """
    Stands in for a ROS node, handing out one prepared service client.
    """

    client: ServiceAnsweringOneOutcome
    """
    The client every request is made through.
    """

    def create_client(self, service_type: Type[Any], service_name: str):
        return self.client


def test_a_sensor_that_was_tared_needs_nothing_further():
    client = ServiceAnsweringOneOutcome(outcome=RetareOutcome.APPLIED)

    WrenchCompensationClient(
        node=NodeOfferingOneService(client=client), service=RETARE_SERVICE
    ).retare()

    assert client.calls == 1


@pytest.mark.parametrize("outcome", [RetareOutcome.MOVED, RetareOutcome.TIMED_OUT])
def test_a_refused_window_stops_the_plan_instead_of_pressing_on(outcome):
    """
    Pressing with an untared sensor drives the tool by the weight of the gripper.
    """
    client = ServiceAnsweringOneOutcome(outcome=outcome)

    with pytest.raises(ForceTorqueSensorNotTared) as raised:
        WrenchCompensationClient(
            node=NodeOfferingOneService(client=client), service=RETARE_SERVICE
        ).retare()

    assert raised.value.outcome is outcome
    assert raised.value.service == RETARE_SERVICE


def test_a_compensation_node_that_is_not_running_is_reported():
    client = ServiceAnsweringOneOutcome(outcome=RetareOutcome.APPLIED, offered=False)

    with pytest.raises(WrenchCompensationUnavailable):
        WrenchCompensationClient(
            node=NodeOfferingOneService(client=client), service=RETARE_SERVICE
        ).retare()

    assert client.calls == 0
