"""
Tests for planning the next g-fold of a T-shirt or a pair of trousers simulated in
MuJoCo, from its silhouette alone, wherever it lies on the table and however far it was
folded before.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest
from PIL import Image

from experiments.garment_folding.bench import FoldingBench
from experiments.garment_folding.garments import (
    BenchGarment,
    BenchTShirt,
    BenchTrousers,
)
from experiments.garment_folding.picture import NextFoldPicture
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.folding.garment import (
    GarmentFold,
    GarmentFoldSequence,
    HalfFold,
)
from semantic_digital_twin.folding.garment_state import GarmentFoldStateEstimator
from semantic_digital_twin.folding.geometry import PlaneTransform
from semantic_digital_twin.folding.strategy import GarmentGFoldStrategy
from semantic_digital_twin.folding.t_shirt import (
    SideFold,
    SleeveFold,
    TShirtFoldSequence,
    TShirtSide,
)
from semantic_digital_twin.folding.trousers import (
    LegFold,
    TrousersFoldSequence,
    TrousersSide,
)
from semantic_digital_twin.semantic_annotations.garments import TrousersLandmark

# %% the bench


def laid_out_bench(
    *,
    garment: BenchGarment,
    seed: int,
    turned_over: bool,
    folds_made: list[GarmentFold],
) -> FoldingBench:
    """
    :param garment: The garment on the table.
    :param seed: Seeds where the garment lies on the table and how it is turned.
    :param turned_over: Whether the garment lies on its other face.
    :param folds_made: The folds the garment was folded by before.
    :return: A running bench with the garment at rest on the table.
    """
    random_generator = numpy.random.default_rng(seed)
    bench = FoldingBench(
        garment=garment,
        placement=PlaneTransform(
            translation=random_generator.uniform(-0.1, 0.1, 2),
            rotation_angle=float(random_generator.uniform(-numpy.pi, numpy.pi)),
        ),
        turned_over=turned_over,
        folds_made=folds_made,
    )
    bench.let_garment_settle()
    return bench


def folding_strategy(
    *, garment: BenchGarment, sequence: GarmentFoldSequence
) -> GarmentGFoldStrategy:
    """
    :param garment: The garment folded.
    :param sequence: The fold sequence it is folded along.
    :return: The strategy folding it with grippers that hold cloth within 6.5 cm.
    """
    return GarmentGFoldStrategy(
        state_estimator=GarmentFoldStateEstimator(
            template=garment.template, sequence=sequence
        ),
        gripper_radius=0.065,
    )


@pytest.fixture(scope="module")
def t_shirt() -> BenchTShirt:
    """
    :return: The T-shirt the bench lays out.
    """
    return BenchTShirt()


@pytest.fixture(scope="module")
def t_shirt_strategy(t_shirt: BenchTShirt) -> GarmentGFoldStrategy:
    """
    :return: The strategy folding the T-shirt along its fold sequence.
    """
    return folding_strategy(garment=t_shirt, sequence=TShirtFoldSequence())


@pytest.fixture(scope="module")
def trousers() -> BenchTrousers:
    """
    :return: The trousers the bench lays out.
    """
    return BenchTrousers()


@pytest.fixture(scope="module")
def trousers_strategy(trousers: BenchTrousers) -> GarmentGFoldStrategy:
    """
    :return: The strategy folding the trousers along their fold sequence.
    """
    return folding_strategy(garment=trousers, sequence=TrousersFoldSequence())


# %% spread T-shirts


@pytest.mark.parametrize("turned_over", [False, True])
def test_a_spread_t_shirt_is_folded_first_at_a_sleeve_held_by_its_cuff(
    t_shirt: BenchTShirt, t_shirt_strategy: GarmentGFoldStrategy, turned_over: bool
) -> None:
    """
    Lying spread out anywhere on the table, on either face, the T-shirt is folded at a
    sleeve first, and the grippers reach both ends of that sleeve's cuff.
    """
    bench = laid_out_bench(
        garment=t_shirt, seed=3, turned_over=turned_over, folds_made=[]
    )

    plan = t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    landmarks = bench.landmark_positions
    bench.stop()

    grasp_positions = numpy.array([grasp.grasp_position for grasp in plan.grasps])
    cuffs = [
        numpy.array([landmarks[side.sleeve_top][:2], landmarks[side.sleeve_bottom][:2]])
        for side in TShirtSide
    ]
    farthest_from_a_grasp = [
        numpy.linalg.norm(cuff[:, None] - grasp_positions[None], axis=2)
        .min(axis=1)
        .max()
        for cuff in cuffs
    ]
    assert isinstance(plan.fold, SleeveFold)
    assert min(farthest_from_a_grasp) <= t_shirt_strategy.gripper_radius


# %% T-shirts folded part of the way


@pytest.mark.parametrize(
    "turned_over, folds_made, next_fold_type",
    [
        (False, [SleeveFold(side=TShirtSide.LEFT)], SideFold),
        (
            False,
            [
                SleeveFold(side=TShirtSide.RIGHT),
                SideFold(side=TShirtSide.RIGHT, strip_width_ratio=0.15),
            ],
            SleeveFold,
        ),
        (True, TShirtFoldSequence().folds[:3], SideFold),
        (
            False,
            [
                SleeveFold(side=TShirtSide.LEFT),
                SideFold(side=TShirtSide.LEFT, strip_width_ratio=0.32),
                SleeveFold(side=TShirtSide.RIGHT),
                SideFold(side=TShirtSide.RIGHT, strip_width_ratio=0.32),
            ],
            HalfFold,
        ),
    ],
)
def test_the_planner_carries_on_from_how_far_the_t_shirt_was_folded(
    t_shirt: BenchTShirt,
    t_shirt_strategy: GarmentGFoldStrategy,
    turned_over: bool,
    folds_made: list[GarmentFold],
    next_fold_type: type[GarmentFold],
) -> None:
    """
    A T-shirt folded part of the way, by folds of the sequence at their own widths or at
    others, is recognised as folded that far and folded on with the fold after.
    """
    bench = laid_out_bench(
        garment=t_shirt, seed=5, turned_over=turned_over, folds_made=folds_made
    )

    plan = t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert len(plan.state.garment.folds) == len(folds_made)
    assert type(plan.fold) is next_fold_type


def test_nothing_is_left_to_fold_on_a_folded_t_shirt(
    t_shirt: BenchTShirt, t_shirt_strategy: GarmentGFoldStrategy
) -> None:
    """
    A T-shirt folded by the whole sequence is recognised as folded.
    """
    bench = laid_out_bench(
        garment=t_shirt,
        seed=7,
        turned_over=False,
        folds_made=TShirtFoldSequence().folds,
    )

    plan = t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert plan is None


# %% folds made by the bench's hands


def test_a_sleeve_fold_the_hands_make_is_recognised_as_made(
    t_shirt: BenchTShirt, t_shirt_strategy: GarmentGFoldStrategy
) -> None:
    """
    Once the hands have folded a sleeve as planned, the planner sees that fold made and
    plans the side fold below that sleeve.
    """
    bench = laid_out_bench(garment=t_shirt, seed=3, turned_over=False, folds_made=[])
    sleeve_fold = t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette())

    bench.perform(plan=sleeve_fold)
    plan = t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert plan.state.garment.folds == [sleeve_fold.fold]
    assert plan.fold == SideFold(side=sleeve_fold.fold.side)


# %% trousers


@pytest.mark.parametrize("turned_over", [False, True])
def test_spread_trousers_are_folded_first_at_a_leg_held_at_its_outseam(
    trousers: BenchTrousers,
    trousers_strategy: GarmentGFoldStrategy,
    turned_over: bool,
) -> None:
    """
    Lying spread out anywhere on the table, on either face, the trousers are folded
    first by laying a leg onto the other, and the grippers reach both ends of that leg's
    outseam.
    """
    bench = laid_out_bench(
        garment=trousers, seed=3, turned_over=turned_over, folds_made=[]
    )

    plan = trousers_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    landmarks = bench.landmark_positions
    bench.stop()

    grasp_positions = numpy.array([grasp.grasp_position for grasp in plan.grasps])
    outseam_ends = [
        numpy.array([landmarks[waist][:2], landmarks[cuff][:2]])
        for waist, cuff in [
            (TrousersLandmark.LEFT_WAIST, TrousersLandmark.LEFT_CUFF_OUTSEAM),
            (TrousersLandmark.RIGHT_WAIST, TrousersLandmark.RIGHT_CUFF_OUTSEAM),
        ]
    ]
    farthest_from_a_grasp = [
        numpy.linalg.norm(ends[:, None] - grasp_positions[None], axis=2)
        .min(axis=1)
        .max()
        for ends in outseam_ends
    ]
    assert type(plan.fold) is LegFold
    assert min(farthest_from_a_grasp) <= trousers_strategy.gripper_radius


@pytest.mark.parametrize(
    "turned_over, folded_leg", [(False, TrousersSide.LEFT), (True, TrousersSide.RIGHT)]
)
def test_trousers_with_a_leg_folded_are_folded_in_half_next(
    trousers: BenchTrousers,
    trousers_strategy: GarmentGFoldStrategy,
    turned_over: bool,
    folded_leg: TrousersSide,
) -> None:
    """
    Trousers with either leg on the other, on either face, are recognised as folded that
    far and are folded in half next.
    """
    bench = laid_out_bench(
        garment=trousers,
        seed=5,
        turned_over=turned_over,
        folds_made=[LegFold(side=folded_leg)],
    )

    plan = trousers_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert [type(fold) for fold in plan.state.garment.folds] == [LegFold]
    assert plan.fold == HalfFold()


@pytest.mark.parametrize("lower_part_ratio", [0.4, 0.5])
def test_trousers_folded_in_half_are_recognised_as_folded_that_deep(
    trousers: BenchTrousers,
    trousers_strategy: GarmentGFoldStrategy,
    lower_part_ratio: float,
) -> None:
    """
    Trousers folded in half at their middle or short of it are recognised as folded, by
    a half fold about as deep as it was made.
    """
    bench = laid_out_bench(
        garment=trousers,
        seed=7,
        turned_over=False,
        folds_made=[
            LegFold(side=TrousersSide.LEFT),
            HalfFold(lower_part_ratio=lower_part_ratio),
        ],
    )

    state = trousers_strategy.state_estimator.estimate(
        silhouette=bench.observe_silhouette()
    )
    bench.stop()

    [_, half_fold] = state.garment.folds
    assert state.next_fold is None
    assert half_fold.depth_ratio == pytest.approx(lower_part_ratio, abs=0.05)


def test_the_hands_fold_spread_trousers_until_nothing_is_left_to_fold(
    trousers: BenchTrousers, trousers_strategy: GarmentGFoldStrategy
) -> None:
    """
    The hands fold spread trousers as the planner says, one fold after the other, each
    planned from what the previous one left, until the planner sees them folded.
    """
    bench = laid_out_bench(garment=trousers, seed=3, turned_over=False, folds_made=[])
    folds_made = []

    plan = trousers_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    while plan is not None and len(folds_made) < len(TrousersFoldSequence().folds):
        bench.perform(plan=plan)
        folds_made.append(type(plan.fold))
        plan = trousers_strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert folds_made == [LegFold, HalfFold]
    assert plan is None


# %% pictures


@pytest.mark.skipif(
    not MujocoSimulator.offscreen_rendering_available(),
    reason="MuJoCo cannot render offscreen in this process",
)
def test_a_picture_of_the_next_fold_is_drawn_over_the_view_from_above(
    t_shirt: BenchTShirt, t_shirt_strategy: GarmentGFoldStrategy, tmp_path: Path
) -> None:
    """
    The picture of the next fold is saved as an image of the size it is drawn at.
    """
    bench = laid_out_bench(garment=t_shirt, seed=3, turned_over=False, folds_made=[])
    picture = NextFoldPicture(
        view=bench.capture_top_view(),
        plan=t_shirt_strategy.plan_next_fold(silhouette=bench.observe_silhouette()),
        gripper_radius=t_shirt_strategy.gripper_radius,
    )
    bench.stop()
    path = tmp_path / "next_fold.png"

    picture.save(path=path)

    assert Image.open(path).size == (
        round(picture.width * picture.dots_per_inch),
        round(picture.height * picture.dots_per_inch),
    )
