import os

import pytest

from krrood.entity_query_language.symbol_graph import SymbolGraph


@pytest.fixture(autouse=True)
def cleanup_after_test():
    # Setup: runs before each krrood_test
    SymbolGraph()
    yield
    SymbolGraph().clear()


@pytest.fixture(autouse=True, scope="session")
def cleanup_ros():
    """
    Fixture to ensure that ROS is properly cleaned up after all tests.
    """
    if os.environ.get("ROS_VERSION") == "2":
        import rclpy

        if not rclpy.ok():
            rclpy.init()
    yield
    if os.environ.get("ROS_VERSION") == "2":
        if rclpy.ok():
            rclpy.shutdown()
