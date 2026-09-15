#!/usr/bin/env python3
"""
Offline validation of the force/torque load compensation against a recorded bag.

Replays the raw wrench and the transforms from a bag, reconstructs the sensor
orientation per sample, runs the reading through
:class:`~giskardpy.ros2_tools.wrench_compensation_node.WrenchGravityCompensator`, and
checks that the compensated wrench on the settled samples collapses to the noise floor
of the identification. This exercises the live node's arithmetic on real data with no
robot attached.

Usage::

    python3 validate_wrench_compensation.py [BAG_DIRECTORY]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from typing_extensions import Dict, List, Tuple

from giskardpy.ros2_tools.wrench_compensation_node import (
    WrenchGravityCompensator,
    WrenchTopic,
)
from semantic_digital_twin.robots.hsrb import HSRBWristForceTorqueSensor
from semantic_digital_twin.spatial_types import Quaternion

GRAVITY_FRAME = "base_footprint"
"""Gravity-aligned frame the sensor orientation is resolved against."""

SENSOR_FRAME = "wrist_ft_sensor_frame"
"""Frame the recorded wrench is expressed in."""

TRANSFORM_TOPICS = ("/tf", "/tf_static")
"""Topics the transform chain is rebuilt from."""

SETTLED_GRAVITY_ALIGNMENT = 0.9995
"""Cosine between consecutive gravity directions above which a sample counts as settled,
so samples taken while the arm swings are left out."""

PASSING_FORCE_ROOT_MEAN_SQUARE = 0.6
"""Force residual a correct compensation stays under, in N."""

PASSING_TORQUE_ROOT_MEAN_SQUARE = 0.05
"""Torque residual a correct compensation stays under, in N m."""

Rotation = np.ndarray
TransformKey = Tuple[str, str]


def seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def rotation_of(rotation) -> Rotation:
    return (
        Quaternion(x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w)
        .to_rotation_matrix()
        .to_np()[:3, :3]
    )


def read_bag(bag_directory: str):
    """
    :param bag_directory: Directory of the recorded bag.
    :return: The wrench samples, the time series of moving transforms, and the static
        transforms.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_directory, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    message_type = {
        topic.name: get_message(topic.type)
        for topic in reader.get_all_topics_and_types()
    }

    wrenches: List[Tuple[float, np.ndarray, np.ndarray]] = []
    moving: Dict[TransformKey, List[Tuple[float, Rotation]]] = {}
    static: Dict[TransformKey, Rotation] = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        message = deserialize_message(data, message_type[topic])
        if topic == WrenchTopic.RAW:
            force, torque = message.wrench.force, message.wrench.torque
            wrenches.append(
                (
                    seconds(message.header.stamp),
                    np.array([force.x, force.y, force.z]),
                    np.array([torque.x, torque.y, torque.z]),
                )
            )
        elif topic in TRANSFORM_TOPICS:
            for transform in message.transforms:
                key = (transform.header.frame_id, transform.child_frame_id)
                rotation = rotation_of(transform.transform.rotation)
                if topic == "/tf_static":
                    static[key] = rotation
                else:
                    moving.setdefault(key, []).append(
                        (seconds(transform.header.stamp), rotation)
                    )
    for series in moving.values():
        series.sort(key=lambda sample: sample[0])
    return wrenches, moving, static


def chain_to_gravity_frame(moving, static) -> List[str]:
    """The frames from the sensor up to the gravity frame."""
    parent = {child: par for par, child in list(static) + list(moving)}
    path, current = [SENSOR_FRAME], SENSOR_FRAME
    while current != GRAVITY_FRAME:
        if current not in parent:
            raise RuntimeError(f"no transform chain {SENSOR_FRAME} -> {GRAVITY_FRAME}")
        current = parent[current]
        path.append(current)
    return path


def base_R_sensor_at(moving, static, path: List[str], time: float) -> Rotation:
    """The sensor orientation in the gravity frame at the sample's time."""
    rotation = np.eye(3)
    for child, parent in zip(path[:-1], path[1:]):
        if (parent, child) in static:
            edge = static[(parent, child)]
        else:
            series = moving[(parent, child)]
            times = np.array([sample[0] for sample in series])
            edge = series[int(np.argmin(np.abs(times - time)))][1]
        rotation = edge @ rotation
    return rotation


def main(bag_directory: str) -> int:
    wrenches, moving, static = read_bag(bag_directory)
    path = chain_to_gravity_frame(moving, static)
    print(f"loaded {len(wrenches)} wrench samples")
    print("transform chain:", " <- ".join(path))

    compensator = WrenchGravityCompensator(load=HSRBWristForceTorqueSensor.load)
    forces, torques, previous_gravity = [], [], None
    for time, force_raw, torque_raw in wrenches:
        base_R_sensor = base_R_sensor_at(moving, static, path, time)
        gravity = compensator.gravity_in_sensor(base_R_sensor)
        if previous_gravity is not None:
            alignment = np.dot(gravity, previous_gravity) / (
                np.linalg.norm(gravity) * np.linalg.norm(previous_gravity)
            )
            if alignment < SETTLED_GRAVITY_ALIGNMENT:
                previous_gravity = gravity
                continue
        previous_gravity = gravity
        force, torque = compensator.contact_wrench(
            force_raw, torque_raw, base_R_sensor
        )
        forces.append(force)
        torques.append(torque)

    forces, torques = np.array(forces), np.array(torques)
    force_residual = float(np.sqrt((forces**2).mean()))
    torque_residual = float(np.sqrt((torques**2).mean()))
    print(f"\nsettled samples: {len(forces)}")
    print(
        f"compensated force  RMS = {force_residual:.4f} N  "
        f"(target < {PASSING_FORCE_ROOT_MEAN_SQUARE})"
    )
    print(
        f"compensated torque RMS = {torque_residual:.4f} Nm "
        f"(target < {PASSING_TORQUE_ROOT_MEAN_SQUARE})"
    )

    passed = (
        force_residual < PASSING_FORCE_ROOT_MEAN_SQUARE
        and torque_residual < PASSING_TORQUE_ROOT_MEAN_SQUARE
    )
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    bag = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "suturo/tf_ft_bag_BA")
    sys.exit(main(bag))
