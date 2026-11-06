"""
Exceptions of the T-shirt demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% grasping


@dataclass
class TShirtNotPinned(DataclassException):
    """
    Raised when the gripper closed with no vertex of the T-shirt close enough to it to
    hold.
    """

    t_shirt_name: str
    """
    The name of the T-shirt that was to be held.
    """

    holding_body_name: str
    """
    The name of the body that was to hold the T-shirt.
    """

    reason: str
    """
    What the simulator reported when asked to pin the T-shirt.
    """

    def error_message(self) -> str:
        return (
            f"{self.holding_body_name} holds no part of {self.t_shirt_name}: "
            f"{self.reason}"
        )

    def suggest_correction(self) -> str:
        return (
            "Bring the gripper closer to the T-shirt before closing it, or widen the "
            "radius the T-shirt is pinned within."
        )
