"""
The folds that fold a spread T-shirt, found from where its landmarks lie.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.folding.garment import (
    DepthAdjustableFold,
    GarmentFold,
    GarmentFoldSequence,
    GarmentSide,
    GarmentTemplate,
    HalfFold,
)
from semantic_digital_twin.folding.geometry import FoldSegment
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

# %% the spread T-shirt


class TShirtSide(GarmentSide):
    """
    The sides of a T-shirt, left and right as its wearer sees them.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def hem(self) -> TShirtLandmark:
        """
        Where the hem meets this side's side seam.
        """
        if self is TShirtSide.LEFT:
            return TShirtLandmark.LEFT_HEM
        return TShirtLandmark.RIGHT_HEM

    @property
    def armpit(self) -> TShirtLandmark:
        """
        This side's armpit.
        """
        if self is TShirtSide.LEFT:
            return TShirtLandmark.LEFT_ARMPIT
        return TShirtLandmark.RIGHT_ARMPIT

    @property
    def shoulder(self) -> TShirtLandmark:
        """
        This side's shoulder, where its sleeve begins.
        """
        if self is TShirtSide.LEFT:
            return TShirtLandmark.LEFT_SHOULDER
        return TShirtLandmark.RIGHT_SHOULDER

    @property
    def sleeve_top(self) -> TShirtLandmark:
        """
        Where this side's cuff meets the seam over its sleeve.
        """
        if self is TShirtSide.LEFT:
            return TShirtLandmark.LEFT_SLEEVE_TOP
        return TShirtLandmark.RIGHT_SLEEVE_TOP

    @property
    def sleeve_bottom(self) -> TShirtLandmark:
        """
        Where this side's cuff meets the seam under its sleeve.
        """
        if self is TShirtSide.LEFT:
            return TShirtLandmark.LEFT_SLEEVE_BOTTOM
        return TShirtLandmark.RIGHT_SLEEVE_BOTTOM


@dataclass(kw_only=True)
class TShirtTemplate(GarmentTemplate[TShirtLandmark]):
    """
    A T-shirt lying spread out.
    """

    @property
    def lower_end_middle(self) -> numpy.ndarray:
        """
        The middle of the hem.
        """
        return (
            self.landmark_positions[TShirtLandmark.LEFT_HEM]
            + self.landmark_positions[TShirtLandmark.RIGHT_HEM]
        ) / 2

    @property
    def upper_end_middle(self) -> numpy.ndarray:
        """
        The middle of the neckline, between the two shoulder seams.
        """
        return (
            self.landmark_positions[TShirtLandmark.LEFT_COLLAR]
            + self.landmark_positions[TShirtLandmark.RIGHT_COLLAR]
        ) / 2


# %% folds


@dataclass(kw_only=True)
class SleeveFold(GarmentFold[TShirtTemplate]):
    """
    Folds a sleeve onto the body over the line from its armpit to its shoulder.
    """

    side: TShirtSide
    """
    The side of the sleeve.
    """

    @property
    def description(self) -> str:
        return f"fold the {self.side.value} sleeve onto the body"

    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        positions = template.landmark_positions
        return FoldSegment.through_points(
            first=positions[self.side.armpit],
            second=positions[self.side.shoulder],
            folded_point=positions[self.side.sleeve_top],
        )


@dataclass(kw_only=True)
class SideFold(GarmentFold[TShirtTemplate], DepthAdjustableFold):
    """
    Folds a strip along one side of the body, with whatever lies on it, onto the body,
    over a segment running from the hem to the level of the collar.
    """

    side: TShirtSide
    """
    The side of the strip.
    """

    strip_width_ratio: float = 0.25
    """
    How wide the strip is, as a share of the width of the hem.
    """

    @property
    def description(self) -> str:
        return (
            f"fold the {self.side.value} side in by {self.strip_width_ratio:.0%} "
            f"of the hem's width"
        )

    @property
    def depth_ratio(self) -> float:
        return self.strip_width_ratio

    @classproperty
    def smallest_depth_ratio(cls) -> float:
        """
        A strip narrower than a tenth of the hem's width hardly changes the T-shirt's
        outline.
        """
        return 0.1

    @classproperty
    def largest_depth_ratio(cls) -> float:
        """
        A strip wider than half the hem's width would reach across the T-shirt's middle.
        """
        return 0.5

    def copied_at_depth(self, *, depth_ratio: float) -> SideFold:
        return replace(self, strip_width_ratio=depth_ratio)

    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        hem = template.landmark_positions[self.side.hem]
        across = template.landmark_positions[self.side.opposite.hem] - hem
        start = hem + self.strip_width_ratio * across
        return FoldSegment.through_points(
            first=start,
            second=start + template.upper_end_middle - template.lower_end_middle,
            folded_point=hem,
        )


@dataclass(kw_only=True)
class TShirtFoldSequence(GarmentFoldSequence[TShirtTemplate]):
    """
    The folds that fold a spread T-shirt into a rectangle, after Miller et al., *A
    Geometric Approach to Robotic Laundry Folding* (IJRR 2012): the sleeve and then the
    side strip of one side, the same on the other side, and last the hem onto the
    collar.
    """

    first_side: TShirtSide = TShirtSide.LEFT
    """
    The side folded first.
    """

    strip_width_ratio: float = 0.25
    """
    How wide each side strip is, as a share of the width of the hem.
    """

    @cached_property
    def folds(self) -> list[GarmentFold[TShirtTemplate]]:
        sides = [self.first_side, self.first_side.opposite]
        return [
            fold
            for side in sides
            for fold in (
                SleeveFold(side=side),
                SideFold(side=side, strip_width_ratio=self.strip_width_ratio),
            )
        ] + [HalfFold()]

    def mirrored(self) -> TShirtFoldSequence:
        return replace(self, first_side=self.first_side.opposite)
