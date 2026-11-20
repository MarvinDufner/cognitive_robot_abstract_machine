"""
Streaming a query: the pipeline keeps answering the current query frame after frame
instead of waiting for the next query.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.identifier import BBIdentifier
from robokudo.utils.query import QueryHandler

# %% the stream


@dataclass(kw_only=True)
class QueryStream:
    """
    Whether the pipeline answers the current query frame after frame.
    """

    is_streaming: bool = False
    """
    Whether the next pipeline iteration answers the current query again instead of
    waiting for the next one.
    """

    started_streams: int = 0
    """
    How many streams were started, so that state kept across the frames of a stream
    can tell a new stream from the current one.
    """


# %% behaviours


class StartQueryStream(Behaviour):
    """
    Marks the current query as streaming, so that every following frame answers it,
    and counts the stream on its first frame.
    """

    def __init__(
        self, *, query_stream: QueryStream, name: str = "StartQueryStream"
    ) -> None:
        """
        :param query_stream: The stream shared with :class:`ContinueQueryStream`.
        :param name: The name of the node in the tree.
        """
        super().__init__(name=name)
        self.query_stream = query_stream
        """
        The stream shared with :class:`ContinueQueryStream`.
        """

    def update(self) -> Status:
        """
        Start the stream, counting it if it was not streaming yet.
        """
        if not self.query_stream.is_streaming:
            self.query_stream.started_streams += 1
        self.query_stream.is_streaming = True
        return Status.SUCCESS


class ContinueQueryStream(Behaviour):
    """
    Lets the pipeline answer the streaming query again, and ends the stream once a newer
    query waits, the client cancels the query, acknowledging the cancel, or the goal
    ended.

    .. note:: Place it in a selector before the behaviours that take up the next query,
       which run whenever it fails.
    """

    def __init__(
        self, *, query_stream: QueryStream, name: str = "ContinueQueryStream"
    ) -> None:
        """
        :param query_stream: The stream shared with :class:`StartQueryStream`.
        :param name: The name of the node in the tree.
        """
        super().__init__(name=name)
        self.query_stream = query_stream
        """
        The stream shared with :class:`StartQueryStream`.
        """

    def update(self) -> Status:
        """
        :return: :attr:`Status.SUCCESS` while the streaming query lasts, otherwise
            :attr:`Status.FAILURE`.
        """
        if not self.query_stream.is_streaming:
            return Status.FAILURE
        if QueryHandler.preempt_requested():
            QueryHandler.acknowledge_preempt_request()
            self.query_stream.is_streaming = False
            self.feedback_message = "Query canceled"
            return Status.FAILURE
        if not Blackboard().get(BBIdentifier.QUERY_SERVER).is_active():
            self.query_stream.is_streaming = False
            self.feedback_message = "Query superseded or ended"
            return Status.FAILURE
        self.feedback_message = "Streaming the current query"
        return Status.SUCCESS
