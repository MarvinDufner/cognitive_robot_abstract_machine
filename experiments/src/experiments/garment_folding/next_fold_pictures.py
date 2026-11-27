"""
Pictures of the next fold the g-fold planner decides on, for garments laid out anywhere
on the folding bench, turned over, or folded part of the way at various depths.

Run :func:`main` with MuJoCo able to render offscreen, such as with ``MUJOCO_GL=egl``::

    python -m experiments.garment_folding.next_fold_pictures GARMENT FOLDER [START ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy

from experiments.garment_folding.bench import FoldingBench
from experiments.garment_folding.garments import (
    BenchGarment,
    BenchTShirt,
    BenchTrousers,
)
from experiments.garment_folding.picture import NextFoldPicture
from semantic_digital_twin.folding.garment import GarmentFold, GarmentFoldSequence
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

# %% start states


@dataclass(kw_only=True)
class FoldingStart:
    """
    A state of a garment to plan the next fold from.
    """

    name: str
    """
    What the state is called, as in the name of the picture of it.
    """

    turned_over: bool
    """
    Whether the garment lies on its other face.
    """

    folds_made: list[GarmentFold]
    """
    The folds the garment was folded by before it was laid on the table.
    """


# %% the series


@dataclass(kw_only=True)
class NextFoldPictureSeries:
    """
    A picture of the next fold for each of a set of start states of a garment, laid out
    at random.
    """

    folder: Path
    """
    Where the pictures are saved.
    """

    garment: BenchGarment
    """
    The garment laid out.
    """

    sequence: GarmentFoldSequence
    """
    The fold sequence the garment is folded along.
    """

    starts: list[FoldingStart]
    """
    The states the garment is laid out in.
    """

    gripper_radius: float = 0.065
    """
    How far from its centre a gripper holds cloth, in metres: half the 13 cm the HSR's
    hand opens.
    """

    seed: int = 0
    """
    Seeds the random placements of the garment.
    """

    largest_offset: float = 0.1
    """
    How far from the middle of the table the garment's middle may lie along each axis,
    in metres.
    """

    def draw(self, *, start: FoldingStart) -> Path:
        """
        Lay the garment out in a start state, anywhere on the table, plan the next fold
        from its silhouette, and save a picture of it.

        :param start: The start state.
        :return: The picture's file.
        """
        random_generator = numpy.random.default_rng([self.seed, *start.name.encode()])
        bench = FoldingBench(
            garment=self.garment,
            placement=PlaneTransform(
                translation=random_generator.uniform(
                    -self.largest_offset, self.largest_offset, 2
                ),
                rotation_angle=float(random_generator.uniform(-numpy.pi, numpy.pi)),
            ),
            turned_over=start.turned_over,
            folds_made=start.folds_made,
        )
        bench.let_garment_settle()
        strategy = GarmentGFoldStrategy(
            state_estimator=GarmentFoldStateEstimator(
                template=self.garment.template, sequence=self.sequence
            ),
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


# %% the garments


class PicturedGarment(StrEnum):
    """
    The garments a series of pictures is drawn for.
    """

    T_SHIRT = "t_shirt"
    """
    A T-shirt spread out, turned over, and folded part of the way, by folds of the
    sequence and by folds narrower and wider than the sequence's.
    """

    TROUSERS = "trousers"
    """
    A pair of trousers spread out, turned over, and with either leg folded.
    """

    def picture_series(self, *, folder: Path) -> NextFoldPictureSeries:
        """
        :param folder: Where the pictures are saved.
        :return: The series of pictures of this garment.
        """
        if self is PicturedGarment.TROUSERS:
            return NextFoldPictureSeries(
                folder=folder,
                garment=BenchTrousers(),
                sequence=TrousersFoldSequence(),
                starts=[
                    FoldingStart(name="spread", turned_over=False, folds_made=[]),
                    FoldingStart(
                        name="spread_turned_over", turned_over=True, folds_made=[]
                    ),
                    FoldingStart(
                        name="left_leg_folded",
                        turned_over=False,
                        folds_made=[LegFold(side=TrousersSide.LEFT)],
                    ),
                    FoldingStart(
                        name="right_leg_folded_turned_over",
                        turned_over=True,
                        folds_made=[LegFold(side=TrousersSide.RIGHT)],
                    ),
                ],
            )
        left, right = TShirtSide.LEFT, TShirtSide.RIGHT
        return NextFoldPictureSeries(
            folder=folder,
            garment=BenchTShirt(),
            sequence=TShirtFoldSequence(),
            starts=[
                FoldingStart(name="spread", turned_over=False, folds_made=[]),
                FoldingStart(
                    name="spread_turned_over", turned_over=True, folds_made=[]
                ),
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
            ],
        )


def main() -> None:
    """
    Draw the pictures of the garment named first on the command line, a
    :class:`PicturedGarment`, of the starts named after the folder, or of every start,
    into the folder.
    """
    series = PicturedGarment(sys.argv[1]).picture_series(folder=Path(sys.argv[2]))
    series.folder.mkdir(parents=True, exist_ok=True)
    names = sys.argv[3:]
    for start in series.starts:
        if not names or start.name in names:
            print(series.draw(start=start))


if __name__ == "__main__":
    main()
