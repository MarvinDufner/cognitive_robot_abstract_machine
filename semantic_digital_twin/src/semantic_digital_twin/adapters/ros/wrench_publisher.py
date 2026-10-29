"""
Where a force/torque sensor's live reading goes while no real sensor publishes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from geometry_msgs.msg import WrenchStamped
from rclpy.publisher import Publisher

from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor

# %% the topic a simulated reading is published on


class SimulatedWrenchTopic(StrEnum):
    """
    The topics readings that no driver produces are published on.
    """

    MEASURED = "simulated_wrench"
    """
    What a simulated sensor measures, in its own frame.
    """


# %% publishing what a sensor reads


@dataclass(eq=False)
class MeasuredWrenchPublisher(StateChangeCallback, HasROS2Node):
    """
    Publishes the live reading of a force/torque sensor whenever the world state
    changes, so readings a simulation writes can be followed the way a real sensor's
    topic is.
    """

    sensor: ForceTorqueSensor = field(kw_only=True)
    """
    The sensor whose reading is published.
    """

    topic: str = field(default=SimulatedWrenchTopic.MEASURED, kw_only=True)
    """
    Topic the readings are published on.
    """

    publisher: Publisher = field(init=False, repr=False)
    """
    Publisher of the readings.
    """

    def __post_init__(self):
        super().__post_init__()
        self.publisher = self.node.create_publisher(WrenchStamped, self.topic, 10)

    def on_state_change(self, **kwargs) -> None:
        """
        Publish what the sensor holds, or nothing while it has read nothing.
        """
        if not self.sensor.has_received_wrench:
            return
        message = WrenchStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = str(self.sensor.root.name)
        force = self.sensor.measured_force
        torque = self.sensor.measured_torque
        message.wrench.force.x = float(force[0])
        message.wrench.force.y = float(force[1])
        message.wrench.force.z = float(force[2])
        message.wrench.torque.x = float(torque[0])
        message.wrench.torque.y = float(torque[1])
        message.wrench.torque.z = float(torque[2])
        self.publisher.publish(message)

    def stop(self) -> None:
        self.node.destroy_publisher(self.publisher)
        super().stop()
