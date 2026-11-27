"""
What goes wrong on the folding bench.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from krrood.exceptions import DataclassException

# %% the hands


@dataclass(kw_only=True)
class TooFewHands(DataclassException):
    """
    Raised when a fold needs more grippers than the bench has hands.
    """

    grasp_count: int
    """
    How many grippers the fold needs.
    """

    hand_count: int
    """
    How many hands the bench has.
    """

    def error_message(self) -> str:
        return (
            f"The fold needs {self.grasp_count} grippers, but the bench has only "
            f"{self.hand_count} hands."
        )

    def suggest_correction(self) -> str:
        return (
            "plan the fold with a larger gripper radius, or give the bench more hands"
        )


@dataclass(kw_only=True)
class NothingToPinch(DataclassException):
    """
    Raised when a hand closes where no cloth lies within its reach.
    """

    position: numpy.ndarray
    """
    Where the hand closed, in the world frame.
    """

    def error_message(self) -> str:
        return f"No vertex of the T-shirt lies within reach of {self.position}."

    def suggest_correction(self) -> str:
        return "grasp the T-shirt where its cloth lies, or widen the pinch radius"
