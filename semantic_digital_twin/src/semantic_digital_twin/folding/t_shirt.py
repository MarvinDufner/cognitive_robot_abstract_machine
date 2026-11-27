"""
The folds that fold a spread T-shirt, found from where its landmarks lie, and the
T-shirt part way through them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import numpy

from semantic_digital_twin.folding.g_fold import LayeredCloth
from semantic_digital_twin.folding.geometry import FoldSegment, Polygon
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

# %% the spread T-shirt


class TShirtSide(Enum):
    """
    The sides of a T-shirt, left and right as its wearer sees them.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def opposite(self) -> TShirtSide:
        """
        The other side.
        """
        return TShirtSide.RIGHT if self is TShirtSide.LEFT else TShirtSide.LEFT

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


@dataclass(kw_only=True)
class TShirtTemplate:
    """
    A T-shirt lying spread out, given by where its landmarks lie in a frame of its own.
    """

    landmark_positions: dict[TShirtLandmark, numpy.ndarray]
    """
    Where each landmark lies, on the plane the T-shirt lies on.
    """

    @cached_property
    def outline(self) -> Polygon:
        """
        The outline of the T-shirt, through its landmarks in the order they follow it.
        """
        return Polygon(
            vertices=numpy.array(
                [self.landmark_positions[landmark] for landmark in TShirtLandmark]
            )
        )

    @property
    def hem_middle(self) -> numpy.ndarray:
        """
        The middle of the hem.
        """
        return (
            self.landmark_positions[TShirtLandmark.LEFT_HEM]
            + self.landmark_positions[TShirtLandmark.RIGHT_HEM]
        ) / 2

    @property
    def collar_middle(self) -> numpy.ndarray:
        """
        The middle of the neckline, between the two shoulder seams.
        """
        return (
            self.landmark_positions[TShirtLandmark.LEFT_COLLAR]
            + self.landmark_positions[TShirtLandmark.RIGHT_COLLAR]
        ) / 2


# %% folds


@dataclass(kw_only=True)
class TShirtFold(ABC):
    """
    One fold of a T-shirt.
    """

    @abstractmethod
    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        """
        :param template: The T-shirt, spread out.
        :return: The segment the fold folds over, in the template's frame. Folding over
            it once the folds before it are made folds the T-shirt as this fold does, as
            long as it crosses only cloth the folds before it left where it was.
        """


@dataclass(kw_only=True)
class SleeveFold(TShirtFold):
    """
    Folds a sleeve onto the body over the line from its armpit to its shoulder.
    """

    side: TShirtSide
    """
    The side of the sleeve.
    """

    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        positions = template.landmark_positions
        return FoldSegment.through_points(
            first=positions[self.side.armpit],
            second=positions[self.side.shoulder],
            folded_point=positions[self.side.sleeve_top],
        )


@dataclass(kw_only=True)
class SideFold(TShirtFold):
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

    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        hem = template.landmark_positions[self.side.hem]
        across = template.landmark_positions[self.side.opposite.hem] - hem
        start = hem + self.strip_width_ratio * across
        return FoldSegment.through_points(
            first=start,
            second=start + template.collar_middle - template.hem_middle,
            folded_point=hem,
        )


@dataclass(kw_only=True)
class HalfFold(TShirtFold):
    """
    Folds the hem's end of the T-shirt onto its collar's end, over a segment across the
    whole T-shirt.
    """

    hem_part_ratio: float = 0.5
    """
    Where the segment crosses the T-shirt, as the share of the distance from the hem's
    middle to the collar's middle that lies on the hem's side of it.
    """

    def fold_segment(self, *, template: TShirtTemplate) -> FoldSegment:
        along = template.collar_middle - template.hem_middle
        middle = template.hem_middle + self.hem_part_ratio * along
        across = numpy.array([-along[1], along[0]]) / numpy.linalg.norm(along)
        half_length = numpy.ptp(template.outline.vertices @ across)
        return FoldSegment.through_points(
            first=middle - half_length * across,
            second=middle + half_length * across,
            folded_point=template.hem_middle,
        )


@dataclass(kw_only=True)
class TShirtFoldSequence:
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
    def folds(self) -> list[TShirtFold]:
        """
        The folds, in the order they are made.
        """
        sides = [self.first_side, self.first_side.opposite]
        return [
            fold
            for side in sides
            for fold in (
                SleeveFold(side=side),
                SideFold(side=side, strip_width_ratio=self.strip_width_ratio),
            )
        ] + [HalfFold()]


# %% folded T-shirts


@dataclass(kw_only=True)
class FoldedTShirt:
    """
    A T-shirt part way through a fold sequence, in its template's frame.
    """

    template: TShirtTemplate
    """
    The T-shirt, spread out.
    """

    folds: list[TShirtFold]
    """
    The folds made to it, in the order they were made.
    """

    @cached_property
    def fold_segments(self) -> list[FoldSegment]:
        """
        The segment of each fold, in the order the folds were made.
        """
        return [fold.fold_segment(template=self.template) for fold in self.folds]

    @cached_property
    def cloth(self) -> LayeredCloth:
        """
        The layers of the folded T-shirt.
        """
        cloth = LayeredCloth(layers=[self.template.outline])
        for segment in self.fold_segments:
            cloth = cloth.folded(segment=segment)
        return cloth

    @cached_property
    def landmark_positions(self) -> dict[TShirtLandmark, numpy.ndarray]:
        """
        Where each landmark of the folded T-shirt lies.
        """
        cloth = LayeredCloth(layers=[self.template.outline])
        positions = numpy.array(
            [self.template.landmark_positions[landmark] for landmark in TShirtLandmark]
        )
        for segment in self.fold_segments:
            positions = cloth.folded_points(points=positions, segment=segment)
            cloth = cloth.folded(segment=segment)
        return dict(zip(TShirtLandmark, positions))
