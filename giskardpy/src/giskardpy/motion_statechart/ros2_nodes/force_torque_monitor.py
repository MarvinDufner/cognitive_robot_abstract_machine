from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from cramph.data_types import ObservationStateValues
from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.monitors.force_monitors import ForceThresholdMonitor
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode


@dataclass(eq=False, repr=False, kw_only=True)
class ForceTorqueNode(TopicSubscriberNode[WrenchStamped]):
    """
    Superclass for all nodes that subscribe to a ROS topic that contains force and
    torque data.
    """

    msg_type: WrenchStamped = field(init=False, default=WrenchStamped)

    def force_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.force.x,
                self.current_msg.wrench.force.y,
                self.current_msg.wrench.force.z,
            ]
        )

    def force_magnitude(self) -> float:
        return float(np.linalg.norm(self.force_as_np()))

    def torque_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.torque.x,
                self.current_msg.wrench.torque.y,
                self.current_msg.wrench.torque.z,
            ]
        )

    def torque_magnitude(self) -> float:
        return float(np.linalg.norm(self.torque_as_np()))


@dataclass(eq=False, repr=False, kw_only=True)
class ForceImpactMonitor(ForceThresholdMonitor, ForceTorqueNode):
    """
    Watches the force published on a topic, for a wrench that is not a sensor the robot
    annotates.
    """

    _observed_msg: WrenchStamped | None = field(init=False, default=None, repr=False)
    """
    The message this node last took a reading from.
    """

    def on_tick(self) -> ObservationStateValues | None:
        ForceTorqueNode.on_tick(self)
        return ForceThresholdMonitor.on_tick(self)

    def new_force(self) -> np.ndarray | None:
        if self.current_msg is self._observed_msg:
            return None
        self._observed_msg = self.current_msg
        return self.force_as_np()

    def on_reset(self) -> None:
        ForceTorqueNode.on_reset(self)
        ForceThresholdMonitor.on_reset(self)
        self._observed_msg = None
