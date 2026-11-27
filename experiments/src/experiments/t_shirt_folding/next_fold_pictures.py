"""
Pictures of the next fold the g-fold planner decides on, for T-shirts laid out anywhere
on the folding bench, turned over, or folded part of the way at various depths.

Run as ``python -m experiments.t_shirt_folding.next_fold_pictures FOLDER [START ...]``
with MuJoCo able to render offscreen, such as with ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy

from experiments.t_shirt_folding.bench import FoldingBench
from experiments.t_shirt_folding.picture import NextFoldPicture
from semantic_digital_twin.folding.geometry import PlaneTransform
from semantic_digital_twin.folding.strategy import TShirtGFoldStrategy
from semantic_digital_twin.folding.t_shirt import (
    SideFold,
    SleeveFold,
    TShirtFold,
    TShirtSide,
)
from semantic_digital_twin.folding.t_shirt_state import TShirtFoldStateEstimator

# %% start states


@dataclass(kw_only=True)
class FoldingStart:
    """
    A state of the T-shirt to plan the next fold from.
    """

    name: str
    """
    What the state is called, as in the name of the picture of it.
    """

    turned_over: bool
    """
    Whether the T-shirt lies on its other face.
    """

    folds_made: list[TShirtFold]
    """
    The folds the T-shirt was folded by before it was laid on the table.
    """


# %% the series


@dataclass(kw_only=True)
class NextFoldPictureSeries:
    """
    A picture of the next fold for each of a set of start states drawn at random.
    """

    folder: Path
    """
    Where the pictures are saved.
    """

    gripper_radius: float = 0.065
    """
    How far from its centre a gripper holds cloth, in metres: half the 13 cm the HSR's
    hand opens.
    """

    seed: int = 0
    """
    Seeds the random placements of the T-shirt.
    """

    largest_offset: float = 0.1
    """
    How far from the middle of the table the T-shirt's middle may lie along each axis,
    in metres.
    """

    @cached_property
    def starts(self) -> list[FoldingStart]:
        """
        The T-shirt spread out, turned over, and folded part of the way, by folds of the
        sequence and by folds narrower and wider than the sequence's.
        """
        left, right = TShirtSide.LEFT, TShirtSide.RIGHT
        return [
            FoldingStart(name="spread", turned_over=False, folds_made=[]),
            FoldingStart(name="spread_turned_over", turned_over=True, folds_made=[]),
            FoldingStart(
                name="left_sleeve_folded",
                turned_over=False,
                folds_made=[SleeveFold(side=left)],
            ),
            FoldingStart(
                name="right_side_folded_narrow",
                turned_over=False,
                folds_made=[
                    SleeveFold(side=right),
                    SideFold(side=right, strip_width_ratio=0.15),
                ],
            ),
            FoldingStart(
                name="three_folds_turned_over",
                turned_over=True,
                folds_made=[
                    SleeveFold(side=left),
                    SideFold(side=left),
                    SleeveFold(side=right),
                ],
            ),
            FoldingStart(
                name="both_sides_folded_wide",
                turned_over=False,
                folds_made=[
                    SleeveFold(side=left),
                    SideFold(side=left, strip_width_ratio=0.32),
                    SleeveFold(side=right),
                    SideFold(side=right, strip_width_ratio=0.32),
                ],
            ),
            FoldingStart(
                name="both_sleeves_folded_out_of_order",
                turned_over=False,
                folds_made=[SleeveFold(side=left), SleeveFold(side=right)],
            ),
        ]

    def draw(self, *, start: FoldingStart) -> Path:
        """
        Lay the T-shirt out in a start state, anywhere on the table, plan the next fold
        from its silhouette, and save a picture of it.

        :param start: The start state.
        :return: The picture's file.
        """
        random_generator = numpy.random.default_rng([self.seed, *start.name.encode()])
        bench = FoldingBench(
            placement=PlaneTransform(
                translation=random_generator.uniform(
                    -self.largest_offset, self.largest_offset, 2
                ),
                rotation_angle=float(random_generator.uniform(-numpy.pi, numpy.pi)),
            ),
            turned_over=start.turned_over,
            folds_made=start.folds_made,
        )
        bench.let_t_shirt_settle()
        strategy = TShirtGFoldStrategy(
            state_estimator=TShirtFoldStateEstimator(template=bench.t_shirt_template),
            gripper_radius=self.gripper_radius,
        )
        plan = strategy.plan_next_fold(silhouette=bench.observe_silhouette())
        path = self.folder / f"{start.name}.png"
        NextFoldPicture(
            view=bench.capture_top_view(),
            plan=plan,
            gripper_radius=self.gripper_radius,
        ).save(path=path)
        bench.stop()
        return path


def main() -> None:
    """
    Draw the pictures of the starts named on the command line, or of every start, into
    the folder named first.
    """
    series = NextFoldPictureSeries(folder=Path(sys.argv[1]))
    series.folder.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:]
    for start in series.starts:
        if not names or start.name in names:
            print(series.draw(start=start))


if __name__ == "__main__":
    main()
