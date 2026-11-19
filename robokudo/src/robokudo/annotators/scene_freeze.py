"""
Freezes the scene after a query named ``freeze``, so that its result stays on view until
the next query arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.identifier import BBIdentifier

# %% the freeze


class SceneQueryName(StrEnum):
    """
    The names of queries that act on the scene rather than on objects in it.
    """

    FREEZE = "freeze"
    """
    Answer the query, then keep its result on view until the next query.
    """


@dataclass(kw_only=True)
class SceneFreeze:
    """
    Whether the pipeline holds the scene of the query it answered last.
    """

    is_frozen: bool = False
    """
    Whether the pipeline waits for the next query before reading the camera again.
    """


# %% annotators


class FreezeSceneOnRequest(BaseAnnotator):
    """
    Freezes the scene once a query named :attr:`SceneQueryName.FREEZE` is answered, and
    lets it run after any other query.
    """

    def __init__(
        self, *, scene_freeze: SceneFreeze, name: str = "FreezeSceneOnRequest"
    ) -> None:
        """
        :param scene_freeze: The freeze shared with :class:`HoldFrozenScene`.
        :param name: The name of the node in the tree.
        """
        super().__init__(name)
        self.scene_freeze = scene_freeze
        """
        The freeze shared with :class:`HoldFrozenScene`.
        """

    def update(self) -> Status:
        """
        Freeze the scene if the answered query is named
        :attr:`SceneQueryName.FREEZE`.
        """
        query = self.get_cas().get(CASViews.QUERY)
        self.scene_freeze.is_frozen = query.query == SceneQueryName.FREEZE
        return Status.SUCCESS


class HoldFrozenScene(BaseAnnotator):
    """
    Holds the pipeline before it reads the camera while the scene is frozen, and thaws
    it once the next query arrives, leaving that query to be processed.

    .. note:: Place it after :func:`~robokudo.idioms.pipeline_init`, whose action server
       check waits for a query that arrived but is not processed yet.
    """

    def __init__(
        self, *, scene_freeze: SceneFreeze, name: str = "HoldFrozenScene"
    ) -> None:
        """
        :param scene_freeze: The freeze shared with :class:`FreezeSceneOnRequest`.
        :param name: The name of the node in the tree.
        """
        super().__init__(name)
        self.scene_freeze = scene_freeze
        """
        The freeze shared with :class:`FreezeSceneOnRequest`.
        """

    def update(self) -> Status:
        """
        :return: :attr:`Status.RUNNING` while the scene is frozen and no query waits,
            otherwise :attr:`Status.SUCCESS`.
        """
        if not self.scene_freeze.is_frozen:
            return Status.SUCCESS
        if Blackboard().get(BBIdentifier.QUERY_SERVER).new_query is None:
            self.feedback_message = "Scene frozen until the next query"
            return Status.RUNNING
        self.scene_freeze.is_frozen = False
        return Status.SUCCESS
