#!/usr/bin/env python3
"""
ROS 2 node and utilities to turn a raw force/torque reading into the external contact
wrench, by removing the weight of what is mounted past the sensor and the sensor's own
bias.

Complements :mod:`giskardpy.ros2_tools.force_torque_filter_node`, which denoises a
signal but deliberately does not model the load: its offset estimator tracks the signal,
so a sustained contact force would slowly be absorbed into the offset.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from geometry_msgs.msg import WrenchStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from typing_extensions import List, Optional, Tuple, Type

from giskardpy.middleware.ros2.exceptions import (
    ForceTorqueSensorNotTared,
    SensorNeedsNoCompensationError,
    WrenchCompensationUnavailable,
)
from semantic_digital_twin.robots.robot_parts import (
    ForceTorqueSensor,
    ForceTorqueSensorLoad,
)
from semantic_digital_twin.spatial_types import Quaternion


class RetareOutcome(StrEnum):
    """
    What a re-tare request settled on.
    """

    TIMED_OUT = "re-tare timed out waiting for samples"
    MOVED = "the sensor was not held still, so the bias is unchanged"
    APPLIED = "bias applied"


# %% compensation


@dataclass
class WrenchGravityCompensator:
    """
    Removes the load's weight, the sensor's constant bias and a per-session re-tare bias
    from a raw wrench, leaving the external contact wrench.
    """

    load: ForceTorqueSensorLoad = field(default_factory=ForceTorqueSensorLoad)
    """
    The static load mounted past the sensor.
    """

    gravity: float = 9.81
    """
    Gravitational acceleration, in m/s^2.
    """

    bias_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """
    Residual force bias measured by the last re-tare, in N.
    """

    bias_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """
    Residual torque bias measured by the last re-tare, in N m.
    """

    def gravity_in_sensor(self, base_R_sensor: np.ndarray) -> np.ndarray:
        """
        :param base_R_sensor: Rotation from sensor to a gravity-aligned frame, whose z
            axis points up.
        :return: The gravity vector expressed in the sensor frame.
        """
        return base_R_sensor.T @ np.array([0.0, 0.0, -self.gravity])

    def without_load(
        self,
        force_raw: np.ndarray,
        torque_raw: np.ndarray,
        base_R_sensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The reading with the load's weight and the constant bias removed, but before the
        re-tare bias. This is the residual a re-tare averages.

        :param force_raw: Raw force in the sensor frame.
        :param torque_raw: Raw torque in the sensor frame.
        :param base_R_sensor: Rotation from sensor to a gravity-aligned frame.
        """
        gravity_in_sensor = self.gravity_in_sensor(base_R_sensor)
        force = force_raw - self.load.force_offset - self.load.mass * gravity_in_sensor
        torque = (
            torque_raw
            - self.load.torque_offset
            - np.cross(self.load.first_moment, gravity_in_sensor)
        )
        return force, torque

    def contact_wrench(
        self,
        force_raw: np.ndarray,
        torque_raw: np.ndarray,
        base_R_sensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The external contact wrench in the sensor frame.

        :param force_raw: Raw force in the sensor frame.
        :param torque_raw: Raw torque in the sensor frame.
        :param base_R_sensor: Rotation from sensor to a gravity-aligned frame.
        """
        force, torque = self.without_load(force_raw, torque_raw, base_R_sensor)
        return force - self.bias_force, torque - self.bias_torque


@dataclass
class BiasEstimator:
    """
    Averages load-compensated residuals collected while the sensor is held still,
    producing the per-session bias to subtract.
    """

    required_samples: int = 100
    """
    How many samples a re-tare window holds.
    """

    stationary_force_spread: float = 0.5
    """
    Largest per-axis force spread a still window may show, in N.

    Above the sensor's own noise, so a window it sat quietly through is accepted, and
    below the movement or contact it has to reject.
    """

    _forces: List[np.ndarray] = field(default_factory=list, init=False)
    """
    Force residuals collected so far.
    """

    _torques: List[np.ndarray] = field(default_factory=list, init=False)
    """
    Torque residuals collected so far.
    """

    def reset(self) -> None:
        """
        Discard the samples collected so far.
        """
        self._forces.clear()
        self._torques.clear()

    @property
    def collected(self) -> int:
        """
        How many residuals the window holds so far.
        """
        return len(self._forces)

    def add(self, force: np.ndarray, torque: np.ndarray) -> bool:
        """
        Add one residual.

        :param force: Load-compensated force residual.
        :param torque: Load-compensated torque residual.
        :return: Whether the window is now full.
        """
        self._forces.append(force)
        self._torques.append(torque)
        return len(self._forces) >= self.required_samples

    @property
    def force_spread(self) -> float:
        """
        The largest per-axis spread of the collected forces, in N.

        What tells a window the sensor sat still through from one it was moved or
        touched during. ``nan`` while nothing has been collected.
        """
        if not self._forces:
            return float("nan")
        return float(np.array(self._forces).std(axis=0).max())

    @property
    def force_shift(self) -> float:
        """
        How far the largest axis moved from the first half of the window to the second,
        in N.

        Separates a reading that is drifting from one that is merely noisy: noise averages
        out between the halves, a drift does not. ``nan`` while the window is too short to
        halve.
        """
        if len(self._forces) < 2:
            return float("nan")
        forces = np.array(self._forces)
        half = len(forces) // 2
        return float(
            np.abs(forces[half:].mean(axis=0) - forces[:half].mean(axis=0)).max()
        )

    def result(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        :return: The mean bias, or None while the window is short or if the sensor moved
            during it.
        """
        if len(self._forces) < self.required_samples:
            return None
        if self.force_spread > self.stationary_force_spread:
            return None
        forces = np.array(self._forces)
        return forces.mean(axis=0), np.array(self._torques).mean(axis=0)


# %% ros node


@dataclass(eq=False)
class WrenchCompensationNode(Node):
    """
    Subscribes a raw wrench, removes the load and bias using the live sensor orientation
    from tf, and republishes the external contact wrench.

    The output stays in the sensor frame, so a consumer that rotates it itself keeps
    working unchanged.
    """

    sensor: Type[ForceTorqueSensor] = field(kw_only=True)
    """
    The sensor to compensate.

    Every name this node uses comes from its annotation.
    """

    queue_depth: int = 1
    """
    Shallow, so a stalled consumer cannot build up latency.
    """

    estimator: BiasEstimator = field(default_factory=BiasEstimator)
    """
    Collects and averages the residuals a re-tare is taken from.
    """

    retare_timeout: float = 5.0
    """
    How long a re-tare waits for its window to fill, in seconds.
    """

    def __post_init__(self) -> None:
        """
        :raises SensorNeedsNoCompensationError: If the sensor's driver already publishes
            the contact wrench, or if compensating it would feed the node its own
            output.
        """
        super().__init__("wrench_compensation")
        if (
            self.sensor.compensation is None
            or self.sensor.compensation.uncompensated_topic_name
            == self.sensor.topic_name
        ):
            raise SensorNeedsNoCompensationError(sensor_type=self.sensor)
        self.compensator = WrenchGravityCompensator(load=self.sensor.compensation.load)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._lock = threading.Lock()
        self._collecting = False
        self._retare_done = threading.Event()
        self._retare_outcome = RetareOutcome.TIMED_OUT

        # A reliable publisher is accepted by either kind of subscriber; subscribing
        # best-effort accepts either kind of sensor.
        subscription_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.queue_depth,
        )
        publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.queue_depth,
        )
        # One reentrant group lets the subscription keep delivering while the blocking
        # re-tare service waits for its window.
        group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(
            WrenchStamped,
            self.sensor.compensation.uncompensated_topic_name,
            self._on_message,
            subscription_qos,
            callback_group=group,
        )
        self.publisher = self.create_publisher(
            WrenchStamped, self.sensor.topic_name, publisher_qos
        )
        self.retare_service = self.create_service(
            Trigger,
            self.sensor.compensation.retare_service,
            self._on_retare,
            callback_group=group,
        )

    def _base_R_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        """
        :param sensor_frame: Frame the incoming wrench is expressed in.
        :return: Rotation from the sensor to the gravity frame, or None while tf cannot
            answer.
        """
        # The latest available transform: the wrist reorients slowly compared to the
        # sensor rate, and asking for the latest avoids extrapolating while it moves.
        gravity_frame = self.sensor.compensation.gravity_frame
        if not self.tf_buffer.can_transform(gravity_frame, sensor_frame, Time()):
            self.get_logger().warn(
                f"tf {gravity_frame} <- {sensor_frame} is not available yet",
                throttle_duration_sec=2.0,
            )
            return None
        transform = self.tf_buffer.lookup_transform(gravity_frame, sensor_frame, Time())
        rotation = transform.transform.rotation
        return Quaternion(
            x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w
        ).rotation_matrix.to_np()[:3, :3]

    def _on_message(self, message: WrenchStamped) -> None:
        base_R_sensor = self._base_R_sensor(
            message.header.frame_id or self.sensor.frame_name
        )
        if base_R_sensor is None:
            return
        force_raw = np.array(
            [message.wrench.force.x, message.wrench.force.y, message.wrench.force.z]
        )
        torque_raw = np.array(
            [message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z]
        )
        force, torque = self.compensator.without_load(
            force_raw, torque_raw, base_R_sensor
        )

        with self._lock:
            if self._collecting and self.estimator.add(force, torque):
                self._finish_retare()

        compensated = WrenchStamped()
        compensated.header = message.header
        contact_force = force - self.compensator.bias_force
        contact_torque = torque - self.compensator.bias_torque
        (
            compensated.wrench.force.x,
            compensated.wrench.force.y,
            compensated.wrench.force.z,
        ) = map(float, contact_force)
        (
            compensated.wrench.torque.x,
            compensated.wrench.torque.y,
            compensated.wrench.torque.z,
        ) = map(float, contact_torque)
        self.publisher.publish(compensated)

    def _on_retare(self, request: Trigger.Request, response: Trigger.Response):
        """
        Tare the sensor for this session: hold it still in free space, then call this.

        Blocks until the window fills or times out.
        """
        with self._lock:
            self.estimator.reset()
            self._retare_done.clear()
            self._collecting = True
        if not self._retare_done.wait(self.retare_timeout):
            with self._lock:
                collected = self.estimator.collected
                self._collecting = False
            # The answer stays exactly the outcome, which the caller parses; how short the
            # window fell only a reader of the log can act on.
            self.get_logger().warn(
                f"re-tare collected {collected} of {self.estimator.required_samples} readings in "
                f"{self.retare_timeout} s: "
                f"{self.sensor.compensation.uncompensated_topic_name} has to "
                f"arrive at "
                f"{self.estimator.required_samples / self.retare_timeout:.0f} Hz or faster, and only "
                f"readings whose sensor frame tf can place are counted."
            )
            response.success = False
            response.message = RetareOutcome.TIMED_OUT
            return response
        response.success = self._retare_outcome is RetareOutcome.APPLIED
        response.message = self._retare_outcome
        return response

    def _finish_retare(self) -> None:
        """
        Apply the collected window.

        Called while holding the lock.
        """
        self._collecting = False
        result = self.estimator.result()
        if result is None:
            self._retare_outcome = RetareOutcome.MOVED
            self.get_logger().warn(
                f"the forces spread by {self.estimator.force_spread:.2f} N over the "
                f"window, more than the "
                f"{self.estimator.stationary_force_spread:.2f} N a sensor at "
                f"rest may, and moved {self.estimator.force_shift:.2f} N from its first "
                f"half to its second. A shift near the spread means the reading is "
                f"drifting rather than noisy, and no bias taken now stays valid; a shift "
                f"near nothing means the spread is this sensor's own noise, and the "
                f"spread belongs above it."
            )
        else:
            self.compensator.bias_force, self.compensator.bias_torque = result
            self._retare_outcome = RetareOutcome.APPLIED
        self.get_logger().info(f"re-tare: {self._retare_outcome}")
        self._retare_done.set()


# %% asking for a re-tare


@dataclass
class WrenchCompensationClient:
    """
    Asks the node that compensates a sensor to re-tare it.
    """

    node: Node
    """
    ROS node the request is made from.
    """

    service: str
    """
    Re-tare service of the node that compensates the sensor.
    """

    service_timeout: float = 5.0
    """
    How long to wait for the service to appear, in s.
    """

    def retare(self) -> None:
        """
        Tare the sensor, so that what it reads afterwards is contact alone.

        The sensor has to hang free and be held still while this runs: the node refuses a
        window it saw move.

        :raises WrenchCompensationUnavailable: If nothing offers the service.
        :raises ForceTorqueSensorNotTared: If the node refuses the window.
        """
        client = self.node.create_client(Trigger, self.service)
        if not client.wait_for_service(timeout_sec=self.service_timeout):
            raise WrenchCompensationUnavailable(service=self.service)
        response = client.call(Trigger.Request())
        if not response.success:
            raise ForceTorqueSensorNotTared(
                service=self.service, outcome=RetareOutcome(response.message)
            )
