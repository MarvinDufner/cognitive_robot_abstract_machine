"""
Tests for planning the next g-fold of a T-shirt simulated in MuJoCo, from its silhouette
alone, wherever it lies on the table and however far it was folded before.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest
from PIL import Image

from experiments.t_shirt_folding.bench import FoldingBench
from experiments.t_shirt_folding.picture import NextFoldPicture
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.folding.geometry import PlaneTransform
from semantic_digital_twin.folding.strategy import TShirtGFoldStrategy
from semantic_digital_twin.folding.t_shirt import (
    HalfFold,
    SideFold,
    SleeveFold,
    TShirtFold,
    TShirtFoldSequence,
    TShirtSide,
)
from semantic_digital_twin.folding.t_shirt_state import TShirtFoldStateEstimator
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

# %% the bench


def laid_out_bench(
    *, seed: int, turned_over: bool, folds_made: list[TShirtFold]
) -> FoldingBench:
    """
    :param seed: Seeds where the T-shirt lies on the table and how it is turned.
    :param turned_over: Whether the T-shirt lies on its other face.
    :param folds_made: The folds the T-shirt was folded by before.
    :return: A running bench with the T-shirt at rest on the table.
    """
    random_generator = numpy.random.default_rng(seed)
    bench = FoldingBench(
        placement=PlaneTransform(
            translation=random_generator.uniform(-0.1, 0.1, 2),
            rotation_angle=float(random_generator.uniform(-numpy.pi, numpy.pi)),
        ),
        turned_over=turned_over,
        folds_made=folds_made,
    )
    bench.let_t_shirt_settle()
    return bench


def folding_strategy(*, bench: FoldingBench) -> TShirtGFoldStrategy:
    """
    :param bench: The bench whose T-shirt is folded.
    :return: The strategy folding it with grippers that hold cloth within 6.5 cm.
    """
    return TShirtGFoldStrategy(
        state_estimator=TShirtFoldStateEstimator(template=bench.t_shirt_template),
        gripper_radius=0.065,
    )


# %% spread T-shirts


@pytest.mark.parametrize("turned_over", [False, True])
def test_a_spread_t_shirt_is_folded_first_at_a_sleeve_held_by_its_cuff(
    turned_over: bool,
) -> None:
    """
    Lying spread out anywhere on the table, on either face, the T-shirt is folded at a
    sleeve first, and the grippers reach both ends of that sleeve's cuff.
    """
    bench = laid_out_bench(seed=3, turned_over=turned_over, folds_made=[])
    strategy = folding_strategy(bench=bench)

    plan = strategy.plan_next_fold(silhouette=bench.observe_silhouette())
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
    assert min(farthest_from_a_grasp) <= strategy.gripper_radius


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
    turned_over: bool, folds_made: list[TShirtFold], next_fold_type: type[TShirtFold]
) -> None:
    """
    A T-shirt folded part of the way, by folds of the sequence at their own widths or at
    others, is recognised as folded that far and folded on with the fold after.
    """
    bench = laid_out_bench(seed=5, turned_over=turned_over, folds_made=folds_made)

    plan = folding_strategy(bench=bench).plan_next_fold(
        silhouette=bench.observe_silhouette()
    )
    bench.stop()

    assert len(plan.state.t_shirt.folds) == len(folds_made)
    assert type(plan.fold) is next_fold_type


def test_nothing_is_left_to_fold_on_a_folded_t_shirt() -> None:
    """
    A T-shirt folded by the whole sequence is recognised as folded.
    """
    bench = laid_out_bench(
        seed=7, turned_over=False, folds_made=TShirtFoldSequence().folds
    )

    plan = folding_strategy(bench=bench).plan_next_fold(
        silhouette=bench.observe_silhouette()
    )
    bench.stop()

    assert plan is None


# %% folds made by the bench's hands


def test_a_sleeve_fold_the_hands_make_is_recognised_as_made() -> None:
    """
    Once the hands have folded a sleeve as planned, the planner sees that fold made and
    plans the side fold below that sleeve.
    """
    bench = laid_out_bench(seed=3, turned_over=False, folds_made=[])
    strategy = folding_strategy(bench=bench)
    sleeve_fold = strategy.plan_next_fold(silhouette=bench.observe_silhouette())

    bench.perform(plan=sleeve_fold)
    plan = strategy.plan_next_fold(silhouette=bench.observe_silhouette())
    bench.stop()

    assert plan.state.t_shirt.folds == [sleeve_fold.fold]
    assert plan.fold == SideFold(side=sleeve_fold.fold.side)


# %% pictures


@pytest.mark.skipif(
    not MujocoSimulator.offscreen_rendering_available(),
    reason="MuJoCo cannot render offscreen in this process",
)
def test_a_picture_of_the_next_fold_is_drawn_over_the_view_from_above(
    tmp_path: Path,
) -> None:
    """
    The picture of the next fold is saved as an image of the size it is drawn at.
    """
    bench = laid_out_bench(seed=3, turned_over=False, folds_made=[])
    strategy = folding_strategy(bench=bench)
    picture = NextFoldPicture(
        view=bench.capture_top_view(),
        plan=strategy.plan_next_fold(silhouette=bench.observe_silhouette()),
        gripper_radius=strategy.gripper_radius,
    )
    bench.stop()
    path = tmp_path / "next_fold.png"

    picture.save(path=path)

    assert Image.open(path).size == (
        round(picture.width * picture.dots_per_inch),
        round(picture.height * picture.dots_per_inch),
    )
