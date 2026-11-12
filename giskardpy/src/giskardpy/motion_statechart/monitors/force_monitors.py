from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from cramph.data_types import ObservationStateValues

from cramph.node import SucceedsOnObservingTrue

from giskardpy.motion_statechart.graph_node import (
    MotionNodeArtifacts,
    MotionStatechartNode,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import Body

# %% watching a tip for contact


@dataclass(eq=False, repr=False, kw_only=True)
class ContactForceReached(SucceedsOnObservingTrue, MotionStatechartNode):
    """
    Observes ``True`` once the force/torque sensor measuring for a link reports a force
    above :attr:`threshold`.

    Reads the sensor annotation rather than a topic, so the same node serves a real
    sensor and one a simulation writes.
    """

    tip_link: Body
    """
    Link whose contact is watched, measured by the sensor closest to it on the chain.
    """

    threshold: float = 2.0
    """
    Force magnitude that counts, in N.

    Above the noise of carrying the tool and below the force a press holds.
    """

    confirming_readings: int = 3
    """
    How many readings in a row have to be above :attr:`threshold`, so that an isolated
    spike in the sensor's data path cannot end a descent in mid-air.
    """

    _sensor: ForceTorqueSensor | None = field(init=False, default=None, repr=False)
    """
    Sensor resolved from :attr:`tip_link`.
    """

    _readings_counted: int = field(init=False, default=0, repr=False)
    """
    How many readings the sensor had written when this node last took one.
    """

    _readings_above_threshold: int = field(init=False, default=0, repr=False)
    """
    How many of the most recent readings in a row were above :attr:`threshold`.
    """

    _observation: ObservationStateValues = field(
        init=False, default=ObservationStateValues.UNKNOWN, repr=False
    )
    """
    What the last reading was observed as, held while no new one arrives.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Resolve the sensor.

        The observation itself is written by :meth:`on_tick`, since the wrench lives
        with the sensor rather than in the controller's own data.
        """
        self._sensor = ForceTorqueSensor.for_tip(self.context.world, self.tip_link)
        return MotionNodeArtifacts()

    def on_tick(self) -> ObservationStateValues | None:
        """
        Counts readings rather than control cycles, which run faster than some sensors
        publish, so that one spike cannot confirm itself.
        """
        if self._sensor.readings_received == self._readings_counted:
            return self._observation
        self._readings_counted = self._sensor.readings_received
        self._observation = self._observe(self._sensor.measured_wrench.force)
        return self._observation

    def _observe(self, force: np.ndarray) -> ObservationStateValues:
        """
        :param force: The force of the reading just taken.
        :return: What it amounts to. Only its magnitude counts: the tool meets a surface
            at whatever angle it is held.
        """
        if np.linalg.norm(force) <= self.threshold:
            self._readings_above_threshold = 0
            return ObservationStateValues.FALSE
        self._readings_above_threshold += 1
        if self._readings_above_threshold >= self.confirming_readings:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_reset(self) -> None:
        self._readings_above_threshold = 0
        self._observation = ObservationStateValues.UNKNOWN
        self._readings_counted = self._sensor.readings_received
