"""
Tests for the folds of a pair of trousers' fold sequence, and for planning them from the
trousers' silhouette, on trousers with straight edges.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.g_fold import GFold
from semantic_digital_twin.folding.garment import FoldedGarment, HalfFold
from semantic_digital_twin.folding.garment_state import GarmentFoldStateEstimator
from semantic_digital_twin.folding.geometry import PlaneTransform
from semantic_digital_twin.folding.silhouette import ClothSilhouette
from semantic_digital_twin.folding.strategy import GarmentGFoldStrategy
from semantic_digital_twin.folding.trousers import (
    LegFold,
    TrousersFoldSequence,
    TrousersSide,
    TrousersTemplate,
)
from semantic_digital_twin.semantic_annotations.garments import TrousersLandmark

# %% single folds


def test_a_leg_fold_lays_the_leg_onto_the_other_leg(
    trousers_template: TrousersTemplate,
) -> None:
    """
    The left leg is mirrored across the line from the waistband's middle to the crotch,
    onto the right leg.
    """
    folded = FoldedGarment(
        template=trousers_template, folds=[LegFold(side=TrousersSide.LEFT)]
    )

    positions = folded.landmark_positions
    for left, right in [
        (TrousersLandmark.LEFT_WAIST, TrousersLandmark.RIGHT_WAIST),
        (TrousersLandmark.LEFT_CUFF_OUTSEAM, TrousersLandmark.RIGHT_CUFF_OUTSEAM),
        (TrousersLandmark.LEFT_CUFF_INSEAM, TrousersLandmark.RIGHT_CUFF_INSEAM),
    ]:
        assert positions[left] == pytest.approx(positions[right])
        assert positions[right] == pytest.approx(
            trousers_template.landmark_positions[right]
        )


def test_a_leg_fold_holds_the_waistband_and_the_cuff_at_the_outseam(
    trousers_template: TrousersTemplate,
) -> None:
    """
    Lifting the left leg, only the ends of its outseam would slide down; the end of its
    inseam hangs from the cuff.
    """
    trousers = FoldedGarment(template=trousers_template, folds=[])
    g_fold = GFold(
        cloth=trousers.cloth,
        segment=LegFold(side=TrousersSide.LEFT).fold_segment(
            template=trousers_template
        ),
    )

    assert sorted(map(tuple, g_fold.vertices_to_hold)) == [(0.2, 1.0), (0.25, 0.0)]


def test_the_half_fold_holds_both_corners_of_the_stacked_cuffs(
    trousers_template: TrousersTemplate,
) -> None:
    """
    With one leg on the other, the cuffs are lifted by their corners, each holding both
    legs, and laid onto the waistband.
    """
    sequence = TrousersFoldSequence()
    trousers = FoldedGarment(template=trousers_template, folds=sequence.folds[:1])
    g_fold = GFold(
        cloth=trousers.cloth,
        segment=sequence.folds[1].fold_segment(template=trousers_template),
    )

    grasps = g_fold.grasps(gripper_radius=0.065)

    assert sorted(map(tuple, (grasp.grasp_position for grasp in grasps))) == [
        (-0.25, 0.0),
        (-0.05, 0.0),
    ]
    assert [len(grasp.held_vertices) for grasp in grasps] == [2, 2]
    assert sorted(map(tuple, (grasp.placement_position for grasp in grasps))) == [
        pytest.approx((-0.25, 1.0)),
        pytest.approx((-0.05, 1.0)),
    ]


# %% the sequence


@pytest.mark.parametrize("folded_leg", list(TrousersSide))
def test_the_fold_sequence_folds_the_trousers_into_a_quarter(
    trousers_template: TrousersTemplate, folded_leg: TrousersSide
) -> None:
    """
    Every layer of the folded trousers lies within the half of the waistband's width on
    the side of the leg that stays, and within the waistband's half of their length.
    """
    sequence = TrousersFoldSequence(folded_leg=folded_leg)
    folded = FoldedGarment(template=trousers_template, folds=sequence.folds)

    vertices = numpy.concatenate([layer.vertices for layer in folded.cloth.layers])
    staying_cuff = trousers_template.landmark_positions[
        folded_leg.opposite.cuff_outseam
    ]
    assert sorted([vertices[:, 0].min(), vertices[:, 0].max()]) == pytest.approx(
        sorted([0.0, staying_cuff[0]])
    )
    assert [vertices[:, 1].min(), vertices[:, 1].max()] == pytest.approx([0.5, 1.0])


def test_the_fold_sequence_folds_a_leg_before_folding_in_half() -> None:
    """
    The folded leg is laid onto the other first, and the cuffs onto the waistband last.
    """
    folds = TrousersFoldSequence(folded_leg=TrousersSide.RIGHT).folds

    assert folds == [LegFold(side=TrousersSide.RIGHT), HalfFold()]


# %% planning from the silhouette


@pytest.fixture(scope="module")
def strategy(trousers_template: TrousersTemplate) -> GarmentGFoldStrategy:
    """
    :return: The strategy folding the trousers with grippers that hold cloth within
        6.5 cm, as wide as the HSR's hand opens.
    """
    return GarmentGFoldStrategy(
        state_estimator=GarmentFoldStateEstimator(
            template=trousers_template, sequence=TrousersFoldSequence()
        ),
        gripper_radius=0.065,
    )


@pytest.fixture
def placement() -> PlaneTransform:
    """
    :return: Where the trousers lie on the table, turned away from their own frame.
    """
    return PlaneTransform(translation=numpy.array([-0.2, 0.3]), rotation_angle=-1.1)


def test_the_strategy_folds_spread_trousers_at_a_leg_first(
    trousers_template: TrousersTemplate,
    strategy: GarmentGFoldStrategy,
    placement: PlaneTransform,
) -> None:
    """
    Spread trousers are folded first by laying a leg onto the other, held at the
    waistband and the cuff.
    """
    silhouette = ClothSilhouette.from_cloth(
        cloth=FoldedGarment(template=trousers_template, folds=[]).cloth.transformed(
            placement
        ),
        cell_size=4e-3,
    )

    plan = strategy.plan_next_fold(silhouette=silhouette)

    assert type(plan.fold) is LegFold
    assert len(plan.grasps) == 2


@pytest.mark.parametrize("folded_leg", list(TrousersSide))
def test_the_strategy_folds_trousers_with_a_leg_folded_in_half(
    trousers_template: TrousersTemplate,
    strategy: GarmentGFoldStrategy,
    placement: PlaneTransform,
    folded_leg: TrousersSide,
) -> None:
    """
    With either leg on the other, the trousers are folded in half over the line across
    their middle.
    """
    sequence = TrousersFoldSequence(folded_leg=folded_leg)
    folded = FoldedGarment(template=trousers_template, folds=sequence.folds[:1])
    silhouette = ClothSilhouette.from_cloth(
        cloth=folded.cloth.transformed(placement), cell_size=4e-3
    )

    plan = strategy.plan_next_fold(silhouette=silhouette)

    expected_segment = (
        sequence.folds[1]
        .fold_segment(template=trousers_template)
        .transformed(placement)
    )
    assert plan.fold == sequence.folds[1]
    assert plan.state.garment.folds == sequence.folds[:1]
    assert numpy.allclose(
        plan.g_fold.segment.normal, expected_segment.normal, atol=0.01
    )
    assert plan.g_fold.segment.signed_distances(
        expected_segment.start
    ) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("lower_part_ratio", [0.4, 0.6])
def test_the_estimator_finds_how_deep_the_trousers_were_folded_in_half(
    trousers_template: TrousersTemplate,
    strategy: GarmentGFoldStrategy,
    placement: PlaneTransform,
    lower_part_ratio: float,
) -> None:
    """
    Trousers folded in half short of their middle or beyond it are recognised as folded,
    by a half fold made as deep as it was made.
    """
    folds = [
        LegFold(side=TrousersSide.LEFT),
        HalfFold(lower_part_ratio=lower_part_ratio),
    ]
    silhouette = ClothSilhouette.from_cloth(
        cloth=FoldedGarment(template=trousers_template, folds=folds).cloth.transformed(
            placement
        ),
        cell_size=4e-3,
    )

    state = strategy.state_estimator.estimate(silhouette=silhouette)

    [_, half_fold] = state.garment.folds
    assert state.next_fold is None
    assert half_fold.depth_ratio == pytest.approx(lower_part_ratio, abs=0.02)
