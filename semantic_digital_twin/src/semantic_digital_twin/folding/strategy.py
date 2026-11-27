"""
Ways to fold cloth lying on a table, each deciding the next fold from what a camera
above the table sees of the cloth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.folding.exceptions import UnrecognisedTShirtState
from semantic_digital_twin.folding.g_fold import FoldGrasp, GFold
from semantic_digital_twin.folding.silhouette import ClothSilhouette
from semantic_digital_twin.folding.t_shirt import TShirtFold
from semantic_digital_twin.folding.t_shirt_state import (
    TShirtFoldState,
    TShirtFoldStateEstimator,
)

# %% plans


@dataclass(kw_only=True)
class FoldPlan:
    """
    The next fold of a cloth lying on a table, and the grasps that make it.
    """

    g_fold: GFold
    """
    The fold, on the table's plane.
    """

    grasps: list[FoldGrasp]
    """
    Where each gripper takes the cloth and the path it carries it along; all of them
    move together.
    """


@dataclass(kw_only=True)
class TShirtFoldPlan(FoldPlan):
    """
    The next fold of a T-shirt's fold sequence.
    """

    state: TShirtFoldState
    """
    How far the T-shirt is folded before the fold, and where it lies.
    """

    fold: TShirtFold
    """
    The fold of the sequence that comes next.
    """


# %% strategies


class FoldingStrategy(ABC):
    """
    A way to fold a cloth lying on a table, one fold at a time.
    """

    @abstractmethod
    def plan_next_fold(self, *, silhouette: ClothSilhouette) -> FoldPlan | None:
        """
        :param silhouette: What a camera looking straight down at the table sees of the
            cloth.
        :return: The fold to make next, ``None`` once the cloth is folded.
        """


@dataclass(kw_only=True)
class TShirtGFoldStrategy(FoldingStrategy):
    """
    Folds a T-shirt by g-folds along its fold sequence, carrying on from however far the
    T-shirt is folded already.
    """

    state_estimator: TShirtFoldStateEstimator
    """
    Finds how far the T-shirt is folded and where it lies.
    """

    gripper_radius: float
    """
    How far from its centre a gripper holds cloth, in metres.
    """

    apex_height_ratio: float = 1.0
    """
    How high each gripper path rises, as a share of its grasp's distance from the fold's
    line; see :attr:`GFold.apex_height_ratio`.
    """

    minimum_overlap: float = 0.8
    """
    The intersection over union with the silhouette seen that a state of the fold
    sequence has to reach for the T-shirt to count as being in that state.
    """

    def plan_next_fold(self, *, silhouette: ClothSilhouette) -> TShirtFoldPlan | None:
        state = self.state_estimator.estimate(silhouette=silhouette)
        if state.overlap < self.minimum_overlap:
            raise UnrecognisedTShirtState(
                overlap=state.overlap, minimum_overlap=self.minimum_overlap
            )
        if state.next_fold is None:
            return None
        g_fold = GFold(
            cloth=state.cloth,
            segment=state.next_fold_segment,
            apex_height_ratio=self.apex_height_ratio,
        )
        return TShirtFoldPlan(
            g_fold=g_fold,
            grasps=g_fold.grasps(gripper_radius=self.gripper_radius),
            state=state,
            fold=state.next_fold,
        )
