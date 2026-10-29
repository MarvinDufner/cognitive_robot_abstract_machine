"""
Publishing the live reading of a force/torque sensor that no driver feeds.
"""

from dataclasses import dataclass, field
from time import sleep

import numpy as np
from geometry_msgs.msg import WrenchStamped
from numpy.testing import assert_allclose
from rclpy.node import Node
from typing_extensions import List, Optional

from semantic_digital_twin.adapters.ros.wrench_publisher import (
    MeasuredWrenchPublisher,
    SimulatedWrenchTopic,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world import World

# %% collecting what was published


@dataclass
class ReceivedWrenches:
    """
    Keeps the messages a subscription has received.
    """

    messages: List[WrenchStamped] = field(default_factory=list)
    """
    The messages, in the order they arrived.
    """

    def __call__(self, message: WrenchStamped) -> None:
        self.messages.append(message)

    def await_message(self, attempts: int = 30) -> Optional[WrenchStamped]:
        """
        :param attempts: How many tenths of a second to wait.
        :return: The first message received, or ``None`` if none arrived.
        """
        for _ in range(attempts):
            if self.messages:
                return self.messages[0]
            sleep(0.1)
        return None


def subscribe(node: Node, publisher: MeasuredWrenchPublisher) -> ReceivedWrenches:
    """
    :return: What the given publisher publishes from now on.
    """
    received = ReceivedWrenches()
    node.create_subscription(
        msg_type=WrenchStamped, topic=publisher.topic, callback=received, qos_profile=10
    )
    # A subscription that is not yet matched misses what is published next.
    sleep(0.5)
    return received


# %% publishing a reading


def test_a_state_change_publishes_what_the_sensor_reads(
    rclpy_node, hsr_world_copy: World
):
    sensor = hsr_world_copy.get_semantic_annotations_by_type(ForceTorqueSensor)[0]
    publisher = MeasuredWrenchPublisher(
        _world=hsr_world_copy, node=rclpy_node, sensor=sensor
    )
    received = subscribe(rclpy_node, publisher)
    sensor.write_wrench(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

    hsr_world_copy.notify_state_change()

    message = received.await_message()
    assert message is not None
    assert message.header.frame_id == str(sensor.root.name)
    force, torque = message.wrench.force, message.wrench.torque
    assert_allclose([force.x, force.y, force.z], sensor.measured_force)
    assert_allclose([torque.x, torque.y, torque.z], sensor.measured_torque)


def test_a_sensor_that_has_read_nothing_publishes_nothing(
    rclpy_node, hsr_world_copy: World
):
    """
    An idle sensor is not a sensor measuring zero, and publishing zero would read as
    contact that was felt and found to be none.
    """
    sensor = hsr_world_copy.get_semantic_annotations_by_type(ForceTorqueSensor)[0]
    publisher = MeasuredWrenchPublisher(
        _world=hsr_world_copy, node=rclpy_node, sensor=sensor
    )
    received = subscribe(rclpy_node, publisher)

    hsr_world_copy.notify_state_change()

    assert received.await_message(attempts=5) is None


def test_the_readings_are_published_on_the_simulated_wrench_topic(
    rclpy_node, hsr_world_copy: World
):
    sensor = hsr_world_copy.get_semantic_annotations_by_type(ForceTorqueSensor)[0]
    publisher = MeasuredWrenchPublisher(
        _world=hsr_world_copy, node=rclpy_node, sensor=sensor
    )

    assert publisher.topic == SimulatedWrenchTopic.MEASURED
