#!/usr/bin/env python3
"""
Run the wrench compensation for one of Tracy's arms.

Each arm carries its own sensor, so each needs its own node. Every name the node uses
comes from that arm's sensor annotation, so nothing has to be repeated here.

Usage::

    python3 tracy_wrench_compensation.py [left|right]
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.executors import MultiThreadedExecutor
from typing_extensions import Dict, Type

from giskardpy.ros2_tools.wrench_compensation_node import WrenchCompensationNode
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.robots.tracy import (
    TracyLeftForceTorqueSensor,
    TracyRightForceTorqueSensor,
)

SENSOR_OF_ARM: Dict[str, Type[ForceTorqueSensor]] = {
    "left": TracyLeftForceTorqueSensor,
    "right": TracyRightForceTorqueSensor,
}
"""
The sensor each arm is named by on the command line.
"""


def main(arm: str = "left") -> None:
    """
    :param arm: Which arm's sensor to compensate.
    """
    if arm not in SENSOR_OF_ARM:
        raise SystemExit(
            f"unknown arm {arm!r}, expected one of {sorted(SENSOR_OF_ARM)}"
        )
    sensor = SENSOR_OF_ARM[arm]

    rclpy.init()
    node = WrenchCompensationNode(
        topic_in=sensor.raw_wrench_topic,
        topic_out=sensor.wrench_topic,
        gravity_frame=sensor.gravity_frame,
        retare_service_name=sensor.retare_service,
        load=sensor.load,
    )
    node.get_logger().info(
        f"compensating {sensor.raw_wrench_topic} -> {sensor.wrench_topic} "
        f"against {sensor.gravity_frame}, re-tare on {sensor.retare_service}"
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(*sys.argv[1:])
