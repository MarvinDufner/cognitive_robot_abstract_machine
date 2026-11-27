"""
The folds that fold a pair of spread trousers, found from where their landmarks lie.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy

from semantic_digital_twin.folding.garment import (
    GarmentFold,
    GarmentFoldSequence,
    GarmentSide,
    GarmentTemplate,
    HalfFold,
)
from semantic_digital_twin.folding.geometry import FoldSegment
from semantic_digital_twin.semantic_annotations.garments import TrousersLandmark

# %% the spread trousers


class TrousersSide(GarmentSide):
    """
    The sides of a pair of trousers, left and right as their wearer sees them.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def cuff_outseam(self) -> TrousersLandmark:
        """
        Where this side's cuff meets its outseam.
        """
        if self is TrousersSide.LEFT:
            return TrousersLandmark.LEFT_CUFF_OUTSEAM
        return TrousersLandmark.RIGHT_CUFF_OUTSEAM


@dataclass(kw_only=True)
class TrousersTemplate(GarmentTemplate[TrousersLandmark]):
    """
    A pair of trousers lying spread out.
    """

    @property
    def lower_end_middle(self) -> numpy.ndarray:
        """
        The middle between the cuffs of both legs.
        """
        return numpy.mean(
            [
                self.landmark_positions[landmark]
                for landmark in (
                    TrousersLandmark.RIGHT_CUFF_OUTSEAM,
                    TrousersLandmark.RIGHT_CUFF_INSEAM,
                    TrousersLandmark.LEFT_CUFF_INSEAM,
                    TrousersLandmark.LEFT_CUFF_OUTSEAM,
                )
            ],
            axis=0,
        )

    @property
    def upper_end_middle(self) -> numpy.ndarray:
        """
        The middle of the waistband.
        """
        return (
            self.landmark_positions[TrousersLandmark.LEFT_WAIST]
            + self.landmark_positions[TrousersLandmark.RIGHT_WAIST]
        ) / 2


# %% folds


@dataclass(kw_only=True)
class LegFold(GarmentFold[TrousersTemplate]):
    """
    Folds one leg, with its half of the seat, onto the other leg, over the line from the
    middle of the waistband to the crotch.
    """

    side: TrousersSide
    """
    The side of the leg folded over.
    """

    @property
    def description(self) -> str:
        return f"fold the {self.side.value} leg onto the {self.side.opposite.value} leg"

    def fold_segment(self, *, template: TrousersTemplate) -> FoldSegment:
        return FoldSegment.through_points(
            first=template.upper_end_middle,
            second=template.landmark_positions[TrousersLandmark.CROTCH],
            folded_point=template.landmark_positions[self.side.cuff_outseam],
        )


@dataclass(kw_only=True)
class TrousersFoldSequence(GarmentFoldSequence[TrousersTemplate]):
    """
    The folds that fold a pair of spread trousers, after Miller et al., *A Geometric
    Approach to Robotic Laundry Folding* (IJRR 2012): one leg onto the other, and then
    the cuffs onto the waistband.
    """

    folded_leg: TrousersSide = TrousersSide.LEFT
    """
    The side of the leg folded onto the other.
    """

    @cached_property
    def folds(self) -> list[GarmentFold[TrousersTemplate]]:
        return [LegFold(side=self.folded_leg), HalfFold()]

    def mirrored(self) -> TrousersFoldSequence:
        return replace(self, folded_leg=self.folded_leg.opposite)
