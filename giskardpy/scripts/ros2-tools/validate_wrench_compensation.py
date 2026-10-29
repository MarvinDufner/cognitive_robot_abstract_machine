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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from typing_extensions import Dict, List, Tuple, Type

from giskardpy.middleware.ros2.exceptions import NoTransformChainInBagError
from giskardpy.ros2_tools.wrench_compensation_node import WrenchGravityCompensator
from semantic_digital_twin.robots.hsrb import HSRBWristForceTorqueSensor
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.spatial_types import Quaternion

Rotation = np.ndarray
TransformKey = Tuple[str, str]


class TransformTopic(StrEnum):
    """
    The topics the transform chain is rebuilt from.
    """

    MOVING = "/tf"
    STATIC = "/tf_static"


def seconds(stamp) -> float:
    """
    :param stamp: A ROS message time stamp.
    :return: The stamp in seconds.
    """
    return stamp.sec + stamp.nanosec * 1e-9


def rotation_of(rotation) -> Rotation:
    """
    :param rotation: A ROS quaternion message.
    :return: The rotation it describes.
    """
    return Quaternion(
        x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w
    ).rotation_matrix.to_np()[:3, :3]


@dataclass
class RecordedTransforms:
    """
    The transforms a bag holds, kept apart by whether they can change.
    """

    moving: Dict[TransformKey, List[Tuple[float, Rotation]]] = field(
        default_factory=dict
    )
    """
    Rotations over time, per parent and child frame.
    """

    static: Dict[TransformKey, Rotation] = field(default_factory=dict)
    """
    Rotations that hold for the whole recording, per parent and child frame.
    """

    def add(self, topic: str, transform) -> None:
        """
        :param topic: The transform topic the message arrived on.
        :param transform: One transform of that message.
        """
        key = (transform.header.frame_id, transform.child_frame_id)
        rotation = rotation_of(transform.transform.rotation)
        if topic == TransformTopic.STATIC:
            self.static[key] = rotation
            return
        self.moving.setdefault(key, []).append(
            (seconds(transform.header.stamp), rotation)
        )

    def chain(self, tip: str, root: str) -> List[str]:
        """
        :param tip: The frame to start at.
        :param root: The frame to end at.
        :return: The frames from ``tip`` up to ``root``.
        :raises NoTransformChainInBagError: If the recording does not connect the two.
        """
        parent = {child: par for par, child in list(self.static) + list(self.moving)}
        path, current = [tip], tip
        while current != root:
            if current not in parent:
                raise NoTransformChainInBagError(tip=tip, root=root)
            current = parent[current]
            path.append(current)
        return path

    def rotation_along(self, path: List[str], time: float) -> Rotation:
        """
        :param path: The frames from the tip up to the root, as :meth:`chain` returns.
        :param time: When the rotation is wanted, in seconds.
        :return: The rotation from the first frame of the path to the last, at the
            recorded sample nearest that time.
        """
        rotation = np.eye(3)
        for child, parent in zip(path[:-1], path[1:]):
            if (parent, child) in self.static:
                edge = self.static[(parent, child)]
            else:
                series = self.moving[(parent, child)]
                times = np.array([sample[0] for sample in series])
                edge = series[int(np.argmin(np.abs(times - time)))][1]
            rotation = edge @ rotation
        return rotation


@dataclass
class CompensationCheck:
    """
    Replays a bag through the compensation of one kind of sensor and reports whether
    what is left is noise.
    """

    sensor: Type[ForceTorqueSensor]
    """
    The sensor whose load, frame and topic the recording is read with.
    """

    settled_gravity_alignment: float = 0.9995
    """
    Cosine between consecutive gravity directions above which a sample counts as
    settled, so samples taken while the arm swings are left out.
    """

    passing_force_root_mean_square: float = 0.6
    """
    Force residual a correct compensation stays under, in N.
    """

    passing_torque_root_mean_square: float = 0.05
    """
    Torque residual a correct compensation stays under, in N m.
    """

    def read_bag(self, bag_directory: str):
        """
        :param bag_directory: Directory of the recorded bag.
        :return: The wrench samples and the transforms recorded alongside them.
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
        transforms = RecordedTransforms()
        while reader.has_next():
            topic, data, _ = reader.read_next()
            message = deserialize_message(data, message_type[topic])
            if topic == self.sensor.compensation.uncompensated_topic_name:
                force, torque = message.wrench.force, message.wrench.torque
                wrenches.append(
                    (
                        seconds(message.header.stamp),
                        np.array([force.x, force.y, force.z]),
                        np.array([torque.x, torque.y, torque.z]),
                    )
                )
            elif topic in tuple(TransformTopic):
                for transform in message.transforms:
                    transforms.add(topic, transform)
        for series in transforms.moving.values():
            series.sort(key=lambda sample: sample[0])
        return wrenches, transforms

    def settled_residuals(self, wrenches, transforms, path):
        """
        :param wrenches: The recorded raw readings.
        :param transforms: The transforms recorded alongside them.
        :param path: The frames from the sensor up to the gravity frame.
        :return: The compensated force and torque of the samples the sensor held still
            for.
        """
        compensator = WrenchGravityCompensator(load=self.sensor.compensation.load)
        forces, torques, previous_gravity = [], [], None
        for time, force_raw, torque_raw in wrenches:
            base_R_sensor = transforms.rotation_along(path, time)
            gravity = compensator.gravity_in_sensor(base_R_sensor)
            if previous_gravity is not None and self._swung(gravity, previous_gravity):
                previous_gravity = gravity
                continue
            previous_gravity = gravity
            force, torque = compensator.contact_wrench(
                force_raw, torque_raw, base_R_sensor
            )
            forces.append(force)
            torques.append(torque)
        return np.array(forces), np.array(torques)

    def _swung(self, gravity: np.ndarray, previous: np.ndarray) -> bool:
        """
        :param gravity: Gravity in the sensor frame at this sample.
        :param previous: The same at the sample before.
        :return: Whether the sensor turned between the two.
        """
        alignment = np.dot(gravity, previous) / (
            np.linalg.norm(gravity) * np.linalg.norm(previous)
        )
        return alignment < self.settled_gravity_alignment

    def run(self, bag_directory: str) -> int:
        """
        :param bag_directory: Directory of the recorded bag.
        :return: Zero if what the compensation leaves is noise, one otherwise.
        """
        wrenches, transforms = self.read_bag(bag_directory)
        path = transforms.chain(
            self.sensor.frame_name, self.sensor.compensation.gravity_frame
        )
        print(f"loaded {len(wrenches)} wrench samples")
        print("transform chain:", " <- ".join(path))

        forces, torques = self.settled_residuals(wrenches, transforms, path)
        force_residual = float(np.sqrt((forces**2).mean()))
        torque_residual = float(np.sqrt((torques**2).mean()))
        print(f"\nsettled samples: {len(forces)}")
        print(
            f"compensated force  RMS = {force_residual:.4f} N  "
            f"(target < {self.passing_force_root_mean_square})"
        )
        print(
            f"compensated torque RMS = {torque_residual:.4f} Nm "
            f"(target < {self.passing_torque_root_mean_square})"
        )

        passed = (
            force_residual < self.passing_force_root_mean_square
            and torque_residual < self.passing_torque_root_mean_square
        )
        print("\nRESULT:", "PASS" if passed else "FAIL")
        return 0 if passed else 1


if __name__ == "__main__":
    bag = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "suturo/tf_ft_bag_BA")
    sys.exit(CompensationCheck(sensor=HSRBWristForceTorqueSensor).run(bag))
