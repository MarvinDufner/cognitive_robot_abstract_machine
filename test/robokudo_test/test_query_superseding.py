"""
Tests for a query that arrives while the query server still processes an earlier one:
the earlier goal ends, and the newer query waits to be processed.
"""

from __future__ import annotations

# %% Imports
import time
from dataclasses import dataclass
from threading import Thread

import pytest
from action_msgs.msg import GoalStatus
from py_trees.blackboard import Blackboard
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robokudo_msgs.action import Query
from typing_extensions import Callable, Iterator

from robokudo.annotators.query import QueryActionServer
from robokudo.identifier import BBIdentifier
from robokudo.utils.query import QueryHandler

# %% Fixtures


@dataclass(kw_only=True)
class QueryConnection:
    """
    A query server and an action client connected to it, both spinning.
    """

    server: QueryActionServer
    """
    The server the queries are sent to.
    """

    client: ActionClient
    """
    The client that sends the queries.
    """

    timeout_seconds: float = 5.0
    """
    How long to wait for the server to react before a test fails.
    """

    def wait_until(self, *, condition: Callable[[], bool]) -> None:
        """
        Wait until the condition holds.

        :param condition: The state the server is expected to reach.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while not condition():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    def send(self, *, query_name: str) -> ClientGoalHandle:
        """
        Send a query and wait until the server takes it up.

        :param query_name: The query's name.
        :return: The handle of the accepted goal.
        """
        future = self.client.send_goal_async(Query.Goal(query=query_name))
        self.wait_until(condition=future.done)
        self.wait_until(
            condition=lambda: (
                self.server.new_query is not None
                and self.server.new_query.query == query_name
            )
        )
        return future.result()

    def wait_for_status(self, *, goal: ClientGoalHandle) -> int:
        """
        :param goal: The goal to wait for.
        :return: The status the goal ended with.
        """
        future = goal.get_result_async()
        self.wait_until(condition=future.done)
        return future.result().status

    def wait_for_result(self, *, goal: ClientGoalHandle) -> Query.Result:
        """
        :param goal: The goal to wait for.
        :return: The result the goal ended with.
        """
        future = goal.get_result_async()
        self.wait_until(condition=future.done)
        return future.result().result


@pytest.fixture
def query_connection(node: Node) -> Iterator[QueryConnection]:
    """
    :return: A spinning query server and an action client connected to it.
    """
    server = QueryActionServer(name="query")
    Blackboard().set(BBIdentifier.QUERY_SERVER, server)
    client = ActionClient(node, Query, "/robokudo/query")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server)
    executor.add_node(node)
    spinning = Thread(target=executor.spin, daemon=True)
    spinning.start()
    connection = QueryConnection(server=server, client=client)
    assert client.wait_for_server(timeout_sec=connection.timeout_seconds)
    yield connection
    if server.query is not None:
        QueryHandler.send_answer(Query.Result())
        connection.wait_until(condition=lambda: server.query is None)
    executor.shutdown()
    client.destroy()
    spinning.join()


# %% Processing


def test_a_query_waiting_to_be_processed_is_not_active(
    query_connection: QueryConnection,
) -> None:
    """
    The server is active only once the pipeline takes the query up.
    """
    query_connection.send(query_name="tshirt-pile")

    assert not query_connection.server.is_active()
    query_connection.server.start_processing()
    assert query_connection.server.is_active()


# %% Superseding


def test_a_newer_query_ends_the_active_one_with_its_latest_result(
    query_connection: QueryConnection,
) -> None:
    """
    The active goal succeeds with the result streamed last, and the newer query waits to
    be processed.
    """
    server = query_connection.server
    active = query_connection.send(query_name="tshirt-pile")
    server.start_processing()
    streamed_result = Query.Result(text_result="latest grasp")
    QueryHandler.send_interim_result(result=streamed_result)

    query_connection.send(query_name="folding-keypoints")

    assert query_connection.wait_for_status(goal=active) == GoalStatus.STATUS_SUCCEEDED
    assert query_connection.wait_for_result(goal=active) == streamed_result
    assert server.new_query.query == "folding-keypoints"
    assert server.query.query == "folding-keypoints"


def test_a_newer_query_aborts_an_active_one_that_streamed_no_result(
    query_connection: QueryConnection,
) -> None:
    """
    Without a streamed result, the active goal has nothing to succeed with.
    """
    active = query_connection.send(query_name="detect")
    query_connection.server.start_processing()

    query_connection.send(query_name="tshirt-pile")

    assert query_connection.wait_for_status(goal=active) == GoalStatus.STATUS_ABORTED
    assert query_connection.server.new_query.query == "tshirt-pile"
