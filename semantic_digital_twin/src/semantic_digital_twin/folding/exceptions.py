"""
What goes wrong while planning how to fold cloth.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% recognising the cloth


@dataclass(kw_only=True)
class UnrecognisedTShirtState(DataclassException):
    """
    Raised when no state of a T-shirt's fold sequence matches the silhouette seen
    closely enough to plan the next fold from it.
    """

    overlap: float
    """
    The intersection over union of the silhouette seen with the one of the state that
    matches it best.
    """

    minimum_overlap: float
    """
    The intersection over union a state has to reach to be recognised.
    """

    def error_message(self) -> str:
        return (
            f"The silhouette seen overlaps every state of the T-shirt's fold sequence "
            f"by at most {self.overlap:.2f} (intersection over union), less than the "
            f"{self.minimum_overlap:.2f} needed."
        )

    def suggest_correction(self) -> str:
        return (
            "spread the T-shirt out flat before folding it, or check that the template "
            "describes this T-shirt"
        )
