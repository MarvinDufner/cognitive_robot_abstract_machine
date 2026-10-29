from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.monitors.force_monitors import (
    ForceThresholdMonitor,
)
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues


@dataclass(eq=False, repr=False)
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


@dataclass(eq=False, repr=False)
class ForceImpactMonitor(ForceThresholdMonitor, ForceTorqueNode):
    """
    Watches the force published on a topic, for a wrench that is not a sensor the robot
    annotates.
    """

    _observed_msg: Optional[WrenchStamped] = field(init=False, default=None, repr=False)
    """
    The message this node last took a reading from.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        ForceTorqueNode.on_tick(self, context)
        return ForceThresholdMonitor.on_tick(self, context)

    def new_force(self) -> Optional[np.ndarray]:
        if self.current_msg is self._observed_msg:
            return None
        self._observed_msg = self.current_msg
        return self.force_as_np()

    def on_reset(self, context: MotionStatechartContext):
        ForceTorqueNode.on_reset(self, context)
        ForceThresholdMonitor.on_reset(self, context)
        self._observed_msg = None
