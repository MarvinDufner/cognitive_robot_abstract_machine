from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import Body

# %% watching a force against a threshold


@dataclass(eq=False, repr=False)
class ForceThresholdMonitor(MotionStatechartNode, ABC):
    """
    Observes ``True`` once the force it watches stays above :attr:`threshold`.

    Subclasses say where the readings come from.
    """

    threshold: float = field(default=2.0, kw_only=True)
    """
    Force magnitude that counts, in N.

    Above the noise of carrying the tool and below the force a press holds.
    """

    confirming_readings: int = field(default=1, kw_only=True)
    """
    How many readings in a row have to be above :attr:`threshold`, so that an isolated
    spike in the sensor's data path cannot end a descent in mid-air.
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

    @abstractmethod
    def new_force(self) -> Optional[np.ndarray]:
        """
        :return: The force of a reading this node has not observed yet, or ``None``
            while there is none.
        """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Counts readings rather than control cycles, which run faster than some sensors
        publish, so that one spike cannot confirm itself.
        """
        force = self.new_force()
        if force is None:
            return self._observation
        self._observation = self._observe(force)
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

    def on_reset(self, context: MotionStatechartContext) -> None:
        self._readings_above_threshold = 0
        self._observation = ObservationStateValues.UNKNOWN


# %% from the sensor that measures for a link


@dataclass(eq=False, repr=False)
class ContactForceReached(ForceThresholdMonitor):
    """
    Watches the force/torque sensor measuring for a link.

    Reads the sensor annotation rather than a topic, so the same node serves a real
    sensor and one a simulation writes.
    """

    tip_link: Body = field(kw_only=True)
    """
    Link whose contact is watched, measured by the sensor closest to it on the chain.
    """

    confirming_readings: int = field(default=3, kw_only=True)

    _sensor: Optional[ForceTorqueSensor] = field(init=False, default=None, repr=False)
    """
    Sensor resolved from :attr:`tip_link`.
    """

    _readings_counted: int = field(init=False, default=0, repr=False)
    """
    How many readings the sensor had written when this node last took one.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Resolve the sensor.

        The observation itself is written by :meth:`on_tick`, since the wrench lives
        with the sensor rather than in the controller's own data.
        """
        self._sensor = ForceTorqueSensor.for_tip(context.world, self.tip_link)
        return NodeArtifacts()

    def new_force(self) -> Optional[np.ndarray]:
        if self._sensor.readings_received == self._readings_counted:
            return None
        self._readings_counted = self._sensor.readings_received
        return self._sensor.measured_force

    def on_reset(self, context: MotionStatechartContext) -> None:
        super().on_reset(context)
        self._readings_counted = self._sensor.readings_received
