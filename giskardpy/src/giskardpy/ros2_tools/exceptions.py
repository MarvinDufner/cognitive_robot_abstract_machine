"""
Exceptions raised while a force/torque sensor's readings are compensated, and the
outcomes one of them carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from giskardpy.data_types.exceptions import SetupException
from giskardpy.middleware.ros2.exceptions import ExecutionException

# %% what a re-tare settles on


class RetareOutcome(StrEnum):
    """
    What a re-tare request settled on.
    """

    TIMED_OUT = "re-tare timed out waiting for samples"
    MOVED = "the sensor was not held still, so the bias is unchanged"
    APPLIED = "bias applied"


# %% what can go wrong


@dataclass
class WrenchCompensationUnavailable(SetupException):
    """
    Raised when nothing offers the service that re-tares a force/torque sensor.
    """

    service: str
    """
    The service that went unanswered.
    """

    def error_message(self) -> str:
        return f'No wrench compensation node offers "{self.service}".'

    def suggest_correction(self) -> str:
        return (
            "Start a wrench compensation node for this sensor, reading the topic its "
            "driver publishes."
        )


@dataclass
class ForceTorqueSensorNotTared(ExecutionException):
    """
    Raised when a force/torque sensor could not be tared before it is used.

    Pressing with an untared sensor drives the tool by whatever the reading holds
    besides contact, which is mostly the weight of what hangs past the sensor.
    """

    service: str
    """
    The re-tare service that was asked.
    """

    outcome: RetareOutcome
    """
    What the compensation node settled on instead.
    """

    def error_message(self) -> str:
        return f'Re-taring through "{self.service}" failed: {self.outcome}'

    def suggest_correction(self) -> str:
        return "Hold the sensor still and out of contact, then ask again."


@dataclass
class SensorNeedsNoCompensationError(SetupException):
    """
    Raised when a compensation node is started for a sensor that has nothing to
    compensate, or whose output is the reading it would have to read.
    """

    sensor_type: type
    """
    The sensor the node was started for.
    """

    def error_message(self) -> str:
        return (
            f"{self.sensor_type.__name__} states no wrench compensation, or states one "
            f"that publishes onto the very topic it reads."
        )

    def suggest_correction(self) -> str:
        return (
            "Read the sensor straight off its driver, or give its annotation a "
            "compensation whose uncompensated topic differs from the one the sensor is "
            "read from."
        )


@dataclass
class NoTransformChainInBagError(SetupException):
    """
    Raised when a recording does not connect two frames a replay has to relate.
    """

    tip: str
    """
    The frame the chain was to start at.
    """

    root: str
    """
    The frame the chain was to end at.
    """

    def error_message(self) -> str:
        return f"The recording holds no transform chain {self.tip} -> {self.root}."

    def suggest_correction(self) -> str:
        return (
            "Record /tf and /tf_static alongside the wrench, over a stretch in which "
            "both frames are published."
        )
