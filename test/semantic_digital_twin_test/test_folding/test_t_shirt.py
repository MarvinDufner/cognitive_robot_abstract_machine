"""
Tests for the folds of a T-shirt's fold sequence, on a T-shirt with straight edges.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.g_fold import GFold
from semantic_digital_twin.folding.t_shirt import (
    FoldedTShirt,
    HalfFold,
    SideFold,
    SleeveFold,
    TShirtFoldSequence,
    TShirtSide,
    TShirtTemplate,
)
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

# %% single folds


def test_a_sleeve_fold_lays_the_sleeve_onto_the_body(
    t_shirt_template: TShirtTemplate,
) -> None:
    """
    The sleeve is mirrored across the line from its armpit to its shoulder.
    """
    folded = FoldedTShirt(
        template=t_shirt_template, folds=[SleeveFold(side=TShirtSide.RIGHT)]
    )

    assert numpy.allclose(
        folded.landmark_positions[TShirtLandmark.RIGHT_SLEEVE_TOP], [-0.1, 0.6]
    )
    assert numpy.allclose(
        folded.landmark_positions[TShirtLandmark.RIGHT_HEM],
        t_shirt_template.landmark_positions[TShirtLandmark.RIGHT_HEM],
    )


def test_a_sleeve_fold_holds_the_corners_of_the_cuff(
    t_shirt_template: TShirtTemplate,
) -> None:
    """
    Lifting the sleeve, only the two ends of its cuff would slide down.
    """
    t_shirt = FoldedTShirt(template=t_shirt_template, folds=[])
    g_fold = GFold(
        cloth=t_shirt.cloth,
        segment=SleeveFold(side=TShirtSide.RIGHT).fold_segment(
            template=t_shirt_template
        ),
    )

    assert sorted(map(tuple, g_fold.vertices_to_hold)) == [(-0.4, 0.4), (-0.4, 0.6)]


def test_a_side_fold_holds_the_hem_corner_and_the_shoulder(
    t_shirt_template: TShirtTemplate,
) -> None:
    """
    Lifting the side strip with the sleeve folded onto it, its hem corner and its
    shoulder would slide down; the sleeve's corner at the armpit rests on the strip.
    """
    t_shirt = FoldedTShirt(
        template=t_shirt_template, folds=[SleeveFold(side=TShirtSide.RIGHT)]
    )
    g_fold = GFold(
        cloth=t_shirt.cloth,
        segment=SideFold(side=TShirtSide.RIGHT).fold_segment(template=t_shirt_template),
    )

    grasps = g_fold.grasps(gripper_radius=0.065)

    assert sorted(map(tuple, (grasp.grasp_position for grasp in grasps))) == [
        (-0.25, 0.0),
        (-0.25, 0.6),
    ]


def test_the_half_fold_holds_the_corners_of_the_folded_hem(
    t_shirt_template: TShirtTemplate,
) -> None:
    """
    With both sides folded in, the hem's end is lifted by the corners of the folded hem;
    the corners of the side strips in the middle rest on the body.
    """
    sequence = TShirtFoldSequence()
    t_shirt = FoldedTShirt(template=t_shirt_template, folds=sequence.folds[:4])
    g_fold = GFold(
        cloth=t_shirt.cloth,
        segment=sequence.folds[4].fold_segment(template=t_shirt_template),
    )

    grasps = g_fold.grasps(gripper_radius=0.065)

    assert sorted(map(tuple, (grasp.grasp_position for grasp in grasps))) == [
        (-0.125, 0.0),
        (0.125, 0.0),
    ]
    assert sorted(map(tuple, (grasp.placement_position for grasp in grasps))) == [
        pytest.approx((-0.125, 0.7)),
        pytest.approx((0.125, 0.7)),
    ]


# %% the sequence


@pytest.mark.parametrize("first_side", list(TShirtSide))
def test_the_fold_sequence_folds_the_t_shirt_into_a_rectangle(
    t_shirt_template: TShirtTemplate, first_side: TShirtSide
) -> None:
    """
    Every layer of the folded T-shirt lies within the middle half of the hem's width and
    the collar's half of its length.
    """
    sequence = TShirtFoldSequence(first_side=first_side)
    folded = FoldedTShirt(template=t_shirt_template, folds=sequence.folds)

    vertices = numpy.concatenate([layer.vertices for layer in folded.cloth.layers])
    assert vertices.min(axis=0) == pytest.approx([-0.125, 0.35])
    assert vertices.max(axis=0) == pytest.approx([0.125, 0.7])


def test_the_fold_sequence_folds_the_first_side_before_the_other(
    t_shirt_template: TShirtTemplate,
) -> None:
    """
    The sleeve and the strip of the first side come before those of the other side, and
    the fold in half comes last.
    """
    folds = TShirtFoldSequence(first_side=TShirtSide.RIGHT).folds

    assert [type(fold) for fold in folds] == [
        SleeveFold,
        SideFold,
        SleeveFold,
        SideFold,
        HalfFold,
    ]
    assert [fold.side for fold in folds[:4]] == [
        TShirtSide.RIGHT,
        TShirtSide.RIGHT,
        TShirtSide.LEFT,
        TShirtSide.LEFT,
    ]
