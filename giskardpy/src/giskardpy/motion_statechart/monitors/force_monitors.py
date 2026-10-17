from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class ContactForceReached(MotionStatechartNode):
    """
    Observes ``True`` once the sensor measuring for a link feels a force above
    :attr:`threshold`.

    Reads the force/torque annotation rather than a topic, so the same node serves a
    real sensor and one a simulation writes.
    """

    tip_link: Body = field(kw_only=True)
    """
    Link whose contact is watched, measured by the sensor closest to it on the chain.
    """

    threshold: float = field(default=2.0, kw_only=True)
    """
    Force magnitude that counts as contact, in N.

    Above the noise of carrying the tool and below the force a press holds.
    """

    confirming_readings: int = field(default=3, kw_only=True)
    """
    How many readings in a row have to be above :attr:`threshold`, so that an isolated
    spike in the sensor's data path cannot end a descent in mid-air.
    """

    _readings_above_threshold: int = field(init=False, default=0, repr=False)
    """
    How many of the most recent readings in a row were above :attr:`threshold`.
    """

    _sensor: Optional[ForceTorqueSensor] = field(init=False, default=None, repr=False)
    """
    Sensor resolved from :attr:`tip_link`.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Resolve the sensor.

        The observation itself is written by :meth:`on_tick`, since the wrench lives
        with the sensor rather than in the controller's own data.
        """
        self._sensor = ForceTorqueSensor.for_tip(context.world, self.tip_link)
        return NodeArtifacts()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Only the magnitude counts: the tool meets a surface at whatever angle it is
        held.
        """
        if not self._sensor.has_received_wrench:
            return ObservationStateValues.UNKNOWN
        if np.linalg.norm(self._sensor.measured_force) <= self.threshold:
            self._readings_above_threshold = 0
            return ObservationStateValues.FALSE
        self._readings_above_threshold += 1
        if self._readings_above_threshold >= self.confirming_readings:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_reset(self, context: MotionStatechartContext) -> None:
        self._readings_above_threshold = 0
