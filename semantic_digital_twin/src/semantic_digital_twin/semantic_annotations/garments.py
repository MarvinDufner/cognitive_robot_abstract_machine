"""
Garments, whose cloth has no rigid shape a body could hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody

# %% landmarks


class TShirtLandmark(StrEnum):
    """
    The points that outline a T-shirt lying spread out, as in the T-shirt model of Miller
    et al., *A Geometric Approach to Robotic Laundry Folding* (IJRR 2012). Left and right
    are the wearer's. The members follow the outline from the right hem corner over the
    collar to the left hem corner.
    """

    RIGHT_HEM = "right_hem"
    """
    Where the hem meets the right side seam.
    """

    RIGHT_ARMPIT = "right_armpit"
    """
    Where the right side seam meets the seam under the right sleeve.
    """

    RIGHT_SLEEVE_BOTTOM = "right_sleeve_bottom"
    """
    Where the right cuff meets the seam under the right sleeve.
    """

    RIGHT_SLEEVE_TOP = "right_sleeve_top"
    """
    Where the right cuff meets the seam over the right sleeve.
    """

    RIGHT_SHOULDER = "right_shoulder"
    """
    The point of the right shoulder seam above the right armpit, where the sleeve
    begins.
    """

    RIGHT_COLLAR = "right_collar"
    """
    Where the neckline meets the right shoulder seam.
    """

    LEFT_COLLAR = "left_collar"
    """
    Where the neckline meets the left shoulder seam.
    """

    LEFT_SHOULDER = "left_shoulder"
    """
    The point of the left shoulder seam above the left armpit, where the sleeve begins.
    """

    LEFT_SLEEVE_TOP = "left_sleeve_top"
    """
    Where the left cuff meets the seam over the left sleeve.
    """

    LEFT_SLEEVE_BOTTOM = "left_sleeve_bottom"
    """
    Where the left cuff meets the seam under the left sleeve.
    """

    LEFT_ARMPIT = "left_armpit"
    """
    Where the left side seam meets the seam under the left sleeve.
    """

    LEFT_HEM = "left_hem"
    """
    Where the hem meets the left side seam.
    """


# %% garments


@dataclass(eq=False)
class TShirt(HasRootBody):
    """
    A T-shirt.

    Its root body has no geometry and does not follow the cloth. In a simulation, the
    :class:`~semantic_digital_twin.world_description.cloth.ClothSheet` whose
    :attr:`~semantic_digital_twin.world_description.cloth.ClothSheet.semantic_annotation`
    it is simulates its cloth.
    """
