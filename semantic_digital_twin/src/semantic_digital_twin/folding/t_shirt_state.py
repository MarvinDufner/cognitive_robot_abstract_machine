"""
How far a T-shirt seen lying on a table is folded, and where it lies, found from its
silhouette alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy
from scipy.optimize import minimize

from semantic_digital_twin.folding.g_fold import LayeredCloth
from semantic_digital_twin.folding.geometry import FoldSegment, PlaneTransform
from semantic_digital_twin.folding.silhouette import ClothSilhouette
from semantic_digital_twin.folding.t_shirt import (
    DepthAdjustableFold,
    FoldedTShirt,
    TShirtFold,
    TShirtFoldSequence,
    TShirtSide,
    TShirtTemplate,
)
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

# %% fold states


@dataclass(kw_only=True)
class TShirtFoldState:
    """
    How far a T-shirt lying on a table is folded along a fold sequence, and where it
    lies.
    """

    sequence: TShirtFoldSequence
    """
    The fold sequence the T-shirt is folded along.
    """

    t_shirt: FoldedTShirt
    """
    The T-shirt with the folds of the sequence made so far, in its template's frame.
    """

    placement: PlaneTransform
    """
    Carries the template's frame onto the table's plane.
    """

    overlap: float
    """
    The intersection over union of the silhouette the state predicts with the one seen.
    """

    @property
    def next_fold(self) -> TShirtFold | None:
        """
        The next fold of the sequence, ``None`` once every fold is made.
        """
        folds = self.sequence.folds
        return (
            folds[len(self.t_shirt.folds)]
            if len(self.t_shirt.folds) < len(folds)
            else None
        )

    @property
    def next_fold_segment(self) -> FoldSegment | None:
        """
        The segment of the next fold on the table's plane, ``None`` once every fold is
        made.
        """
        if self.next_fold is None:
            return None
        return self.next_fold.fold_segment(template=self.t_shirt.template).transformed(
            self.placement
        )

    @cached_property
    def cloth(self) -> LayeredCloth:
        """
        The layers of the T-shirt on the table's plane.
        """
        return self.t_shirt.cloth.transformed(self.placement)

    @cached_property
    def landmark_positions(self) -> dict[TShirtLandmark, numpy.ndarray]:
        """
        Where each landmark of the T-shirt lies on the table's plane.
        """
        return {
            landmark: self.placement.apply(position)
            for landmark, position in self.t_shirt.landmark_positions.items()
        }


# %% estimation


@dataclass(kw_only=True)
class _FoldStage:
    """
    A T-shirt with the first folds of a sequence made, and the silhouette it has in its
    template's frame.
    """

    sequence: TShirtFoldSequence
    """
    The fold sequence.
    """

    t_shirt: FoldedTShirt
    """
    The T-shirt with the folds made.
    """

    silhouette: ClothSilhouette
    """
    The silhouette of the T-shirt, in its template's frame.
    """


@dataclass(kw_only=True)
class TShirtFoldStateEstimator:
    """
    Finds how far a T-shirt lying on a table is folded, and where it lies, as the state
    of a fold sequence whose silhouette, placed best, overlaps the one seen most.
    """

    template: TShirtTemplate
    """
    The T-shirt, spread out.
    """

    sequences: list[TShirtFoldSequence] = field(
        default_factory=lambda: [
            TShirtFoldSequence(first_side=side) for side in TShirtSide
        ]
    )
    """
    The fold sequences the T-shirt may be folded along.
    """

    scale_tolerance: float = 0.15
    """
    How much larger or smaller than the template the T-shirt may be, as a share of the
    template's size.
    """

    model_cell_size: float = 2.5e-3
    """
    The length of a side of the cells the silhouette of each state is drawn on, in
    metres.
    """

    sample_size: int = 2000
    """
    How many of the covered cells of the silhouette seen, at least, the overlap is
    estimated from.
    """

    rotation_step: float = numpy.pi / 18
    """
    How far apart the turns lie that the search for a state's placement starts from, in
    radians.
    """

    search_start_count: int = 2
    """
    How many of those turns, the ones overlapping most, the search starts from.
    """

    refined_stage_count: int = 3
    """
    How many of the states that overlap the silhouette seen most after a rough fit are
    fitted precisely, the depths of their folds included.
    """

    scanned_depths: numpy.ndarray = field(
        default_factory=lambda: numpy.linspace(0.05, 0.95, 19)
    )
    """
    The depths, as shares of the lengths they are measured along, that each fold of a
    state is tried at before its placement and its folds' depths are fitted together.
    """

    refinement_cell_size: float = 7.5e-3
    """
    How far apart the points lie, in metres, that the overlap of a precise fit is
    counted on.
    """

    @cached_property
    def _stages(self) -> list[_FoldStage]:
        """
        Every state of every sequence, the spread T-shirt only once.
        """
        stages = []
        for sequence in self.sequences:
            first_fold_count = 0 if not stages else 1
            for fold_count in range(first_fold_count, len(sequence.folds) + 1):
                t_shirt = FoldedTShirt(
                    template=self.template, folds=sequence.folds[:fold_count]
                )
                stages.append(
                    _FoldStage(
                        sequence=sequence,
                        t_shirt=t_shirt,
                        silhouette=ClothSilhouette.from_cloth(
                            cloth=t_shirt.cloth, cell_size=self.model_cell_size
                        ),
                    )
                )
        return stages

    def estimate(self, *, silhouette: ClothSilhouette) -> TShirtFoldState:
        """
        :param silhouette: The silhouette of the T-shirt seen on the table.
        :return: The state that matches the silhouette best.
        """
        covered_points = silhouette.covered_points
        sample = covered_points[:: max(1, len(covered_points) // self.sample_size)]
        roughly_fitted = [
            self._fitted_state(
                stage=stage,
                silhouette=silhouette,
                sample=sample,
                starts=self._centred_placements(
                    stage=stage, silhouette=silhouette, sample=sample
                ),
            )
            for stage in self._stages
        ]
        best_rough_fits = sorted(
            roughly_fitted, key=lambda state: state.overlap, reverse=True
        )[: self.refined_stage_count]
        margin = self.scale_tolerance * float(numpy.ptp(covered_points, axis=0).max())
        lower = covered_points.min(axis=0) - margin
        upper = covered_points.max(axis=0) + margin
        columns, rows = numpy.meshgrid(
            numpy.arange(lower[0], upper[0], self.refinement_cell_size),
            numpy.arange(lower[1], upper[1], self.refinement_cell_size),
        )
        grid_points = numpy.column_stack([columns.ravel(), rows.ravel()])
        return max(
            (
                self._refined_state(
                    state=state,
                    grid_points=grid_points,
                    seen=silhouette.covers(grid_points),
                )
                for state in best_rough_fits
            ),
            key=lambda state: state.overlap,
        )

    def _refined_state(
        self, *, state: TShirtFoldState, grid_points: numpy.ndarray, seen: numpy.ndarray
    ) -> TShirtFoldState:
        """
        :param state: A state placed roughly.
        :param grid_points: Points of a grid on the table's plane around the silhouette
            seen, which the overlap is counted on.
        :param seen: Whether the silhouette seen covers each grid point.
        :return: The state with its placement and the depth of each of its folds that
            may lie deeper or less deep fitted to the silhouette seen.
        """
        folds = state.t_shirt.folds
        adjustable = [
            index
            for index, fold in enumerate(folds)
            if isinstance(fold, DepthAdjustableFold)
        ]

        def t_shirt(depth_ratios: numpy.ndarray) -> FoldedTShirt:
            """
            :param depth_ratios: The depth of each fold that may lie at another depth.
            :return: The T-shirt with its folds at those depths.
            """
            adjusted = list(folds)
            for index, depth_ratio in zip(adjustable, depth_ratios):
                adjusted[index] = folds[index].copied_at_depth(
                    depth_ratio=float(numpy.clip(depth_ratio, 0.0, 1.0))
                )
            return FoldedTShirt(template=self.template, folds=adjusted)

        def missing_overlap(parameters: numpy.ndarray) -> float:
            """
            :param parameters: The placement's parameters, see :meth:`_placement`,
                followed by the depth of each fold that may lie at another depth.
            :return: How far the overlap of the state they describe with the silhouette
                seen falls short of one.
            """
            covered = t_shirt(parameters[4:]).cloth.contains(
                self._placement(parameters[:4]).apply_inverse(grid_points)
            )
            return 1 - (covered & seen).sum() / max((covered | seen).sum(), 1)

        placement = state.placement
        parameters = numpy.array(
            [
                *placement.translation,
                placement.rotation_angle,
                numpy.log(placement.scale),
                *(folds[index].depth_ratio for index in adjustable),
            ]
        )
        for parameter_index in range(4, len(parameters)):
            candidates = numpy.repeat(
                parameters[None], len(self.scanned_depths), axis=0
            )
            candidates[:, parameter_index] = self.scanned_depths
            parameters = min(candidates, key=missing_overlap)
        steps = [0.01, 0.01, 0.05, 0.02] + [0.05] * len(adjustable)
        for _ in range(2):
            parameters = minimize(
                missing_overlap,
                parameters,
                method="Nelder-Mead",
                options={
                    "initial_simplex": numpy.vstack(
                        [parameters, parameters + numpy.diag(steps)]
                    ),
                    "xatol": 1e-3,
                    "fatol": 1e-4,
                },
            ).x
        return TShirtFoldState(
            sequence=state.sequence,
            t_shirt=t_shirt(parameters[4:]),
            placement=self._placement(parameters[:4]),
            overlap=1 - missing_overlap(parameters),
        )

    def _placement(self, parameters: numpy.ndarray) -> PlaneTransform:
        """
        :param parameters: The position, turn and logarithm of the scale of a
            placement.
        :return: The placement they describe, its scale within the tolerance.
        """
        x, y, angle, log_scale = parameters
        return PlaneTransform(
            translation=numpy.array([x, y]),
            rotation_angle=angle,
            scale=float(
                numpy.clip(
                    numpy.exp(log_scale),
                    1 - self.scale_tolerance,
                    1 + self.scale_tolerance,
                )
            ),
        )

    def _centred_placements(
        self, *, stage: _FoldStage, silhouette: ClothSilhouette, sample: numpy.ndarray
    ) -> list[PlaneTransform]:
        """
        :param stage: A state of a fold sequence.
        :param silhouette: The silhouette seen.
        :param sample: The centres of the covered cells of the silhouette seen that the
            overlap is estimated from.
        :return: Of the placements that lay the state's silhouette onto the one seen
            with their centres on each other and their areas as equal as the scale
            tolerance allows, turned by every :attr:`rotation_step`, those that overlap
            most, the best first.
        """
        scale = float(
            numpy.clip(
                numpy.sqrt(silhouette.area / stage.silhouette.area),
                1 - self.scale_tolerance,
                1 + self.scale_tolerance,
            )
        )
        placements = []
        for angle in numpy.arange(0.0, 2 * numpy.pi, self.rotation_step):
            placement = PlaneTransform(
                translation=numpy.zeros(2), rotation_angle=float(angle), scale=scale
            )
            placement.translation = silhouette.centroid - placement.apply(
                stage.silhouette.centroid
            )
            placements.append(placement)
        return sorted(
            placements,
            key=lambda placement: self._overlap(
                stage=stage, silhouette=silhouette, sample=sample, placement=placement
            ),
            reverse=True,
        )[: self.search_start_count]

    @staticmethod
    def _overlap(
        *,
        stage: _FoldStage,
        silhouette: ClothSilhouette,
        sample: numpy.ndarray,
        placement: PlaneTransform,
    ) -> float:
        """
        :param stage: A state of a fold sequence.
        :param silhouette: The silhouette seen.
        :param sample: The centres of the covered cells of the silhouette seen that the
            overlap is estimated from.
        :param placement: Carries the template's frame onto the table's plane.
        :return: The intersection over union of the state's silhouette, placed, with
            the one seen.
        """
        model_area = stage.silhouette.area * placement.scale**2
        intersection = min(
            silhouette.area
            * float(stage.silhouette.covers(placement.apply_inverse(sample)).mean()),
            model_area,
        )
        return intersection / (silhouette.area + model_area - intersection)

    def _fitted_state(
        self,
        *,
        stage: _FoldStage,
        silhouette: ClothSilhouette,
        sample: numpy.ndarray,
        starts: list[PlaneTransform],
    ) -> TShirtFoldState:
        """
        :param stage: A state of a fold sequence.
        :param silhouette: The silhouette seen.
        :param sample: The centres of the covered cells of the silhouette seen that the
            overlap is estimated from.
        :param starts: The placements to search from.
        :return: The state, roughly placed where its silhouette overlaps the one seen
            most.
        """

        def missing_overlap(parameters: numpy.ndarray) -> float:
            """
            :param parameters: The parameters of a placement, see :meth:`_placement`.
            :return: How far the overlap of the placement falls short of one.
            """
            return 1 - self._overlap(
                stage=stage,
                silhouette=silhouette,
                sample=sample,
                placement=self._placement(parameters),
            )

        fits = []
        for start in starts:
            initial = numpy.array(
                [*start.translation, start.rotation_angle, numpy.log(start.scale)]
            )
            fits.append(
                minimize(
                    missing_overlap,
                    initial,
                    method="Nelder-Mead",
                    options={
                        "initial_simplex": numpy.vstack(
                            [initial, initial + numpy.diag([0.03, 0.03, 0.2, 0.05])]
                        ),
                        "xatol": 1e-2,
                        "fatol": 1e-2,
                    },
                )
            )
        best = min(fits, key=lambda fit: fit.fun)
        return TShirtFoldState(
            sequence=stage.sequence,
            t_shirt=stage.t_shirt,
            placement=self._placement(best.x),
            overlap=1 - float(best.fun),
        )
