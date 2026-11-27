"""
Tests for finding how far a T-shirt is folded from its silhouette, and for planning its
next g-fold from that.
"""

from __future__ import annotations

import numpy
import pytest
import trimesh

from semantic_digital_twin.folding.exceptions import UnrecognisedTShirtState
from semantic_digital_twin.folding.geometry import PlaneTransform
from semantic_digital_twin.folding.silhouette import ClothSilhouette
from semantic_digital_twin.folding.strategy import TShirtGFoldStrategy
from semantic_digital_twin.folding.t_shirt import (
    FoldedTShirt,
    TShirtFoldSequence,
    TShirtSide,
    TShirtTemplate,
)
from semantic_digital_twin.folding.t_shirt_state import TShirtFoldStateEstimator

# %% silhouettes


def test_the_silhouette_of_a_surface_covers_its_triangles() -> None:
    """
    A square of two triangles, seen from above, covers its area.
    """
    square = trimesh.Trimesh(
        vertices=numpy.array(
            [[0.0, 0.0, 0.1], [0.5, 0.0, 0.1], [0.5, 0.5, 0.2], [0.0, 0.5, 0.1]]
        ),
        faces=numpy.array([[0, 1, 2], [0, 2, 3]]),
    )

    silhouette = ClothSilhouette.from_surface(surface=square, cell_size=0.01)

    assert silhouette.area == pytest.approx(0.25, rel=0.05)
    assert silhouette.centroid == pytest.approx([0.25, 0.25], abs=0.01)


# %% states


@pytest.fixture(scope="module")
def strategy(t_shirt_template: TShirtTemplate) -> TShirtGFoldStrategy:
    """
    :return: The strategy folding the T-shirt with grippers that hold cloth within
        6.5 cm, as wide as the HSR's hand opens.
    """
    return TShirtGFoldStrategy(
        state_estimator=TShirtFoldStateEstimator(template=t_shirt_template),
        gripper_radius=0.065,
    )


@pytest.fixture
def placement() -> PlaneTransform:
    """
    :return: Where the T-shirt lies on the table, turned away from its own frame.
    """
    return PlaneTransform(translation=numpy.array([0.4, -0.3]), rotation_angle=2.2)


def test_the_strategy_folds_a_spread_t_shirt_starting_with_its_first_sequence(
    t_shirt_template: TShirtTemplate,
    strategy: TShirtGFoldStrategy,
    placement: PlaneTransform,
) -> None:
    """
    Either side of a spread T-shirt can be folded first; the strategy takes the first
    fold of the first sequence its estimator knows.
    """
    silhouette = ClothSilhouette.from_cloth(
        cloth=FoldedTShirt(template=t_shirt_template, folds=[]).cloth.transformed(
            placement
        ),
        cell_size=4e-3,
    )

    plan = strategy.plan_next_fold(silhouette=silhouette)

    assert plan.fold == strategy.state_estimator.sequences[0].folds[0]


@pytest.mark.parametrize("first_side", list(TShirtSide))
@pytest.mark.parametrize("fold_count", range(1, 4))
def test_the_strategy_plans_the_fold_that_comes_next(
    t_shirt_template: TShirtTemplate,
    strategy: TShirtGFoldStrategy,
    placement: PlaneTransform,
    first_side: TShirtSide,
    fold_count: int,
) -> None:
    """
    From the silhouette of a T-shirt with some folds of a sequence made, the strategy
    plans the next fold of that sequence, where the T-shirt lies.
    """
    sequence = TShirtFoldSequence(first_side=first_side)
    folded = FoldedTShirt(template=t_shirt_template, folds=sequence.folds[:fold_count])
    silhouette = ClothSilhouette.from_cloth(
        cloth=folded.cloth.transformed(placement), cell_size=4e-3
    )

    plan = strategy.plan_next_fold(silhouette=silhouette)

    expected_segment = sequence.folds[fold_count].fold_segment(
        template=t_shirt_template
    )
    expected_segment = expected_segment.transformed(placement)
    assert plan.fold == sequence.folds[fold_count]
    assert numpy.allclose(
        plan.g_fold.segment.normal, expected_segment.normal, atol=0.01
    )
    assert plan.g_fold.segment.signed_distances(
        expected_segment.start
    ) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("first_side", list(TShirtSide))
def test_the_strategy_folds_a_t_shirt_with_both_sides_in_across_the_middle(
    t_shirt_template: TShirtTemplate,
    strategy: TShirtGFoldStrategy,
    placement: PlaneTransform,
    first_side: TShirtSide,
) -> None:
    """
    With both sides folded in, the T-shirt is folded in half over the line across its
    middle; seen from above, either end may be the one folded onto the other.
    """
    sequence = TShirtFoldSequence(first_side=first_side)
    folded = FoldedTShirt(template=t_shirt_template, folds=sequence.folds[:4])
    silhouette = ClothSilhouette.from_cloth(
        cloth=folded.cloth.transformed(placement), cell_size=4e-3
    )

    plan = strategy.plan_next_fold(silhouette=silhouette)

    expected_segment = (
        sequence.folds[4].fold_segment(template=t_shirt_template).transformed(placement)
    )
    assert plan.fold == sequence.folds[4]
    assert plan.g_fold.segment.signed_distances(
        numpy.array([expected_segment.start, expected_segment.end])
    ) == pytest.approx([0.0, 0.0], abs=0.01)


def test_the_strategy_plans_nothing_once_the_t_shirt_is_folded(
    t_shirt_template: TShirtTemplate,
    strategy: TShirtGFoldStrategy,
    placement: PlaneTransform,
) -> None:
    """
    After the last fold of the sequence, there is nothing left to fold.
    """
    sequence = TShirtFoldSequence()
    folded = FoldedTShirt(template=t_shirt_template, folds=sequence.folds)
    silhouette = ClothSilhouette.from_cloth(
        cloth=folded.cloth.transformed(placement), cell_size=4e-3
    )

    assert strategy.plan_next_fold(silhouette=silhouette) is None


def test_the_strategy_refuses_a_silhouette_unlike_any_state(
    strategy: TShirtGFoldStrategy,
) -> None:
    """
    A T-shirt twisted into a rope, 0.9 m long and 8 cm wide, matches no state of the
    fold sequence.
    """
    rope = ClothSilhouette(
        covered=numpy.ones((20, 225), dtype=bool),
        origin=numpy.zeros(2),
        cell_size=4e-3,
    )

    with pytest.raises(UnrecognisedTShirtState):
        strategy.plan_next_fold(silhouette=rope)
