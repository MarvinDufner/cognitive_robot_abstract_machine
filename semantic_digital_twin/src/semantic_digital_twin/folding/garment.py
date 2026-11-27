"""
Garments lying spread out, given by where their landmarks lie, the folds that fold them,
and garments part way through a sequence of those folds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from functools import cached_property

import numpy
from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.folding.g_fold import LayeredCloth
from semantic_digital_twin.folding.geometry import FoldSegment, Polygon

# %% the spread garment

LandmarkType = TypeVar("LandmarkType", bound=StrEnum)
"""
The landmarks of the kind of garment a :class:`GarmentTemplate` describes, their members
in the order they follow its outline.
"""


class GarmentSide(Enum):
    """
    The two sides of a kind of garment, left and right as its wearer sees them.
    """

    @property
    def opposite(self) -> Self:
        """
        The other side.
        """
        return next(side for side in type(self) if side is not self)


@dataclass(kw_only=True)
class GarmentTemplate(Generic[LandmarkType], SubClassSafeGeneric, ABC):
    """
    A garment lying spread out, given by where its landmarks lie in a frame of its own.
    """

    landmark_positions: dict[LandmarkType, numpy.ndarray]
    """
    Where each landmark lies, on the plane the garment lies on.
    """

    @cached_property
    def outline(self) -> Polygon:
        """
        The outline of the garment, through its landmarks in the order they follow it.
        """
        landmarks = type(self).get_type_of_generic_parameter(LandmarkType)
        return Polygon(
            vertices=numpy.array(
                [self.landmark_positions[landmark] for landmark in landmarks]
            )
        )

    @property
    @abstractmethod
    def lower_end_middle(self) -> numpy.ndarray:
        """
        The middle of the end of the garment nearer its wearer's feet, such as the
        middle of a T-shirt's hem.
        """

    @property
    @abstractmethod
    def upper_end_middle(self) -> numpy.ndarray:
        """
        The middle of the end of the garment nearer its wearer's head, such as the
        middle of a T-shirt's neckline.
        """


# %% folds

TemplateType = TypeVar("TemplateType", bound=GarmentTemplate)
"""
The kind of spread garment a fold or a fold sequence is worked out on.
"""


@dataclass(kw_only=True)
class GarmentFold(Generic[TemplateType], SubClassSafeGeneric, ABC):
    """
    One fold of a garment.
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        What the fold does, in words, for people reading about it.
        """

    @abstractmethod
    def fold_segment(self, *, template: TemplateType) -> FoldSegment:
        """
        :param template: The garment, spread out.
        :return: The segment the fold folds over, in the template's frame. Folding over
            it once the folds before it are made folds the garment as this fold does, as
            long as it crosses only cloth the folds before it left where it was.
        """


class DepthAdjustableFold(ABC):
    """
    A fold whose segment may lie deeper in the garment or less deep, as folds made by
    hand or by a robot rarely lie exactly where they were planned.
    """

    @property
    @abstractmethod
    def depth_ratio(self) -> float:
        """
        How deep the fold reaches into the garment, as a share of the length of the
        garment it is measured along.
        """

    @abstractmethod
    def copied_at_depth(self, *, depth_ratio: float) -> Self:
        """
        :param depth_ratio: How deep the copy reaches, see :attr:`depth_ratio`.
        :return: The same fold, reaching that deep.
        """


@dataclass(kw_only=True)
class HalfFold(GarmentFold[GarmentTemplate], DepthAdjustableFold):
    """
    Folds the lower end of a garment onto its upper end, over a segment across the
    whole garment.
    """

    lower_part_ratio: float = 0.5
    """
    Where the segment crosses the garment, as the share of the distance from the lower
    end's middle to the upper end's middle that lies on the lower end's side of it.
    """

    @property
    def description(self) -> str:
        return "fold the lower end onto the upper end"

    @property
    def depth_ratio(self) -> float:
        return self.lower_part_ratio

    def copied_at_depth(self, *, depth_ratio: float) -> HalfFold:
        return replace(self, lower_part_ratio=depth_ratio)

    def fold_segment(self, *, template: GarmentTemplate) -> FoldSegment:
        along = template.upper_end_middle - template.lower_end_middle
        middle = template.lower_end_middle + self.lower_part_ratio * along
        across = numpy.array([-along[1], along[0]]) / numpy.linalg.norm(along)
        half_length = numpy.ptp(template.outline.vertices @ across)
        return FoldSegment.through_points(
            first=middle - half_length * across,
            second=middle + half_length * across,
            folded_point=template.lower_end_middle,
        )


@dataclass(kw_only=True)
class GarmentFoldSequence(Generic[TemplateType], SubClassSafeGeneric, ABC):
    """
    The folds that fold a spread garment of one kind, in the order they are made.
    """

    @property
    @abstractmethod
    def folds(self) -> list[GarmentFold[TemplateType]]:
        """
        The folds, in the order they are made.
        """

    @abstractmethod
    def mirrored(self) -> Self:
        """
        :return: The sequence with left and right swapped.
        """


# %% folded garments


@dataclass(kw_only=True)
class FoldedGarment:
    """
    A garment part way through a fold sequence, in its template's frame.
    """

    template: GarmentTemplate
    """
    The garment, spread out.
    """

    folds: list[GarmentFold]
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
        The layers of the folded garment.
        """
        cloth = LayeredCloth(layers=[self.template.outline])
        for segment in self.fold_segments:
            cloth = cloth.folded(segment=segment)
        return cloth

    @cached_property
    def landmark_positions(self) -> dict[StrEnum, numpy.ndarray]:
        """
        Where each landmark of the folded garment lies.
        """
        cloth = LayeredCloth(layers=[self.template.outline])
        positions = numpy.array(list(self.template.landmark_positions.values()))
        for segment in self.fold_segments:
            positions = cloth.folded_points(points=positions, segment=segment)
            cloth = cloth.folded(segment=segment)
        return dict(zip(self.template.landmark_positions, positions))
