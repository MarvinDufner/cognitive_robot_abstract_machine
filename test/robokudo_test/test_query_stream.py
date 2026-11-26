"""
Tests for streaming a query: the pipeline keeps answering the current query frame after
frame, until a newer query arrives, the client cancels it or its goal ends.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from robokudo_msgs.action import Query

from robokudo.behaviours.query_stream import (
    ContinueQueryStream,
    QueryStream,
    StartQueryStream,
)
from robokudo.identifier import BBIdentifier

from .dataset.waiting_query_server import WaitingQueryServer

# %% Fixtures


@pytest.fixture
def query_server() -> WaitingQueryServer:
    """
    :return: A query server whose current query a pipeline took up, with no cancel
        requested.
    """
    server = WaitingQueryServer(query=Query.Goal(query="tshirt-pile"))
    blackboard = Blackboard()
    blackboard.set(BBIdentifier.QUERY_SERVER, server)
    blackboard.set(BBIdentifier.QUERY_PREEMPT_REQUESTED, False)
    blackboard.set(BBIdentifier.QUERY_PREEMPT_ACK, False)
    return server


@pytest.fixture
def streaming() -> QueryStream:
    """
    :return: A stream started for the current query.
    """
    return QueryStream(is_streaming=True)


def update_once(continuation: ContinueQueryStream) -> Status:
    """
    :return: The status the continuation ends its tick with.
    """
    continuation.tick_once()
    return continuation.status


# %% Starting


def test_starting_a_stream_counts_it_once() -> None:
    """
    Each stream is counted when its first frame starts it, not again on later frames, so
    that state kept across a stream's frames can tell a new stream.
    """
    query_stream = QueryStream()
    start = StartQueryStream(query_stream=query_stream)

    start.tick_once()
    start.tick_once()

    assert query_stream.started_streams == 1


def test_starting_the_stream_marks_the_query_as_streaming() -> None:
    """
    The subtree answering a query frame after frame starts the stream.
    """
    query_stream = QueryStream()

    StartQueryStream(query_stream=query_stream).tick_once()

    assert query_stream.is_streaming


# %% Continuing


@pytest.mark.usefixtures("query_server")
def test_without_a_stream_the_pipeline_waits_for_the_next_query() -> None:
    """
    A query answered once leaves the pipeline to take up the next query.
    """
    status = update_once(ContinueQueryStream(query_stream=QueryStream()))

    assert status == Status.FAILURE


@pytest.mark.usefixtures("query_server")
def test_a_stream_continues_while_its_query_lasts(streaming: QueryStream) -> None:
    """
    The next frame answers the same query, without waiting for another.
    """
    status = update_once(ContinueQueryStream(query_stream=streaming))

    assert status == Status.SUCCESS
    assert streaming.is_streaming


# %% Ending


def test_a_newer_query_ends_the_stream_and_stays_waiting(
    query_server: WaitingQueryServer, streaming: QueryStream
) -> None:
    """
    The pipeline goes on to take up the newer query.
    """
    newer_query = Query.Goal(query="folding-keypoints")
    query_server.query = newer_query
    query_server.new_query = newer_query

    status = update_once(ContinueQueryStream(query_stream=streaming))

    assert status == Status.FAILURE
    assert not streaming.is_streaming
    assert query_server.new_query is newer_query


@pytest.mark.usefixtures("query_server")
def test_a_cancel_ends_the_stream_and_is_acknowledged(streaming: QueryStream) -> None:
    """
    Acknowledging the cancel lets the server cancel the goal.
    """
    Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, True)

    status = update_once(ContinueQueryStream(query_stream=streaming))

    assert status == Status.FAILURE
    assert not streaming.is_streaming
    assert Blackboard().get(BBIdentifier.QUERY_PREEMPT_ACK)


def test_an_ended_goal_ends_the_stream(
    query_server: WaitingQueryServer, streaming: QueryStream
) -> None:
    """
    Once the server ended the goal, for instance after an error, nothing is left to
    stream.
    """
    query_server.query = None

    status = update_once(ContinueQueryStream(query_stream=streaming))

    assert status == Status.FAILURE
    assert not streaming.is_streaming
