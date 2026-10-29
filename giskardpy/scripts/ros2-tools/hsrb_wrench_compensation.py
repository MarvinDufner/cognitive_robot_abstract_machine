#!/usr/bin/env python3
"""
Run the wrench compensation for the HSRB's wrist sensor.

Usage::
    python3 hsrb_wrench_compensation.py
"""

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from giskardpy.ros2_tools.wrench_compensation_node import WrenchCompensationNode
from semantic_digital_twin.robots.hsrb import HSRBWristForceTorqueSensor


def main() -> None:
    """
    Compensate the wrist sensor until the process is stopped.
    """
    rclpy.init()
    node = WrenchCompensationNode(sensor=HSRBWristForceTorqueSensor)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
