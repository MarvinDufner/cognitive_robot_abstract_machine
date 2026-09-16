from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Self, List

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
    HasEndEffector,
    HasSensors,
)
from semantic_digital_twin.robots.robot_parts import (
    ForceTorqueSensor,
    AbstractRobot,
    Arm,
    Camera,
    Finger,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class TracyJoint(StrEnum):
    """
    Names of the Tracy's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the grippers' inner knuckle
        and finger tip joints, are left out.
    """

    LEFT_SHOULDER_PAN = "left_shoulder_pan_joint"
    LEFT_SHOULDER_LIFT = "left_shoulder_lift_joint"
    LEFT_ELBOW = "left_elbow_joint"
    LEFT_WRIST_1 = "left_wrist_1_joint"
    LEFT_WRIST_2 = "left_wrist_2_joint"
    LEFT_WRIST_3 = "left_wrist_3_joint"
    LEFT_GRIPPER_LEFT_KNUCKLE = "left_robotiq_85_left_knuckle_joint"
    LEFT_GRIPPER_RIGHT_KNUCKLE = "left_robotiq_85_right_knuckle_joint"

    RIGHT_SHOULDER_PAN = "right_shoulder_pan_joint"
    RIGHT_SHOULDER_LIFT = "right_shoulder_lift_joint"
    RIGHT_ELBOW = "right_elbow_joint"
    RIGHT_WRIST_1 = "right_wrist_1_joint"
    RIGHT_WRIST_2 = "right_wrist_2_joint"
    RIGHT_WRIST_3 = "right_wrist_3_joint"
    RIGHT_GRIPPER_LEFT_KNUCKLE = "right_robotiq_85_left_knuckle_joint"
    RIGHT_GRIPPER_RIGHT_KNUCKLE = "right_robotiq_85_right_knuckle_joint"


@dataclass(eq=False)
class TracyLeftGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_left_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class TracyLeftGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_right_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_right_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class TracyRightGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_left_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class TracyRightGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_right_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_right_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class TracyLeftGripper(
    EndEffector, HasTwoFingers[TracyLeftGripperLeftFinger, TracyLeftGripperRightFinger]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        left_gripper_joints = [
            self._world.get_connection_by_name(TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE),
            self._world.get_connection_by_name(TracyJoint.LEFT_GRIPPER_RIGHT_KNUCKLE),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [
                        0.8,
                        -0.8,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class TracyRightGripper(
    EndEffector,
    HasTwoFingers[TracyRightGripperLeftFinger, TracyRightGripperRightFinger],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        right_gripper_joints = [
            self._world.get_connection_by_name(TracyJoint.RIGHT_GRIPPER_LEFT_KNUCKLE),
            self._world.get_connection_by_name(TracyJoint.RIGHT_GRIPPER_RIGHT_KNUCKLE),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.8, -0.8])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


class TracyWrenchTopic(StrEnum):
    """
    The topics each arm's force/torque reading travels on between the sensor and its
    consumers.
    """

    LEFT_RAW = "/left_arm/force_torque_sensor_broadcaster/wrench"
    """What the left arm's driver publishes, still carrying the weight of the gripper."""

    LEFT_COMPENSATED = "/left_arm/wrench/compensated"
    """The left arm's contact wrench, once the load and the bias have been removed.

    .. todo:: Nothing publishes this yet. A wrench compensation node has to run for the
        left arm, reading :attr:`LEFT_RAW`; until it does, the sensor stays idle.
    """

    RIGHT_RAW = "/right_arm/force_torque_sensor_broadcaster/wrench"
    """What the right arm's driver publishes, still carrying the weight of the gripper."""

    RIGHT_COMPENSATED = "/right_arm/wrench/compensated"
    """The right arm's contact wrench, once the load and the bias have been removed.

    .. todo:: Nothing publishes this yet, as for :attr:`LEFT_COMPENSATED`.
    """


class TracyFrame(StrEnum):
    """
    Frames of Tracy that something outside the kinematic chain has to name.
    """

    LEVEL = "table"
    """The bench the arms are bolted to, and the root of Tracy's own tree: level, so its
    z axis points up."""


class TracyWrenchService(StrEnum):
    """
    The services that zero each arm's wrench compensation.

    Each arm needs its own compensation node, so each has its own service, named after
    the node running in that arm's namespace.
    """

    LEFT = "/left_arm/wrench_compensation/retare"
    RIGHT = "/right_arm/wrench_compensation/retare"


@dataclass(eq=False)
class TracyLeftForceTorqueSensor(ForceTorqueSensor):
    """
    The force/torque sensor in the left wrist.

    Rooted at the flange rather than at the wrist's own ``ft_frame``, because that is the
    frame the driver stamps its readings with; the two are rotated half a turn apart, so
    reading one as the other flips the sign of two axes.

    .. todo:: Identify :attr:`load` for whatever the left gripper carries; it defaults to
        nothing mounted, so the whole reading is taken as contact force, and the gripper
        alone weighs several newtons.
    """

    @classproperty
    def wrench_topic(cls) -> str:
        return TracyWrenchTopic.LEFT_COMPENSATED

    @classproperty
    def raw_wrench_topic(cls) -> str:
        return TracyWrenchTopic.LEFT_RAW

    @classproperty
    def median_readings(cls) -> int:
        """
        Wide enough for the spikes this sensor's data path produces: measured at up to
        12 N against a 0.9 N reading, isolated, and never less than three readings apart
        at the 100 Hz the driver publishes, so a window of five always holds a majority of
        good readings.
        """
        return 5

    @classproperty
    def gravity_frame(cls) -> str:
        return TracyFrame.LEVEL

    @classproperty
    def retare_service(cls) -> str:
        return TracyWrenchService.LEFT

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "left_tool0")
        )


@dataclass(eq=False)
class TracyRightForceTorqueSensor(ForceTorqueSensor):
    """
    The force/torque sensor in the right wrist.

    Rooted at the flange, as for :class:`TracyLeftForceTorqueSensor`.

    .. todo:: Identify :attr:`load` for whatever the right gripper carries.
    """

    @classproperty
    def wrench_topic(cls) -> str:
        return TracyWrenchTopic.RIGHT_COMPENSATED

    @classproperty
    def raw_wrench_topic(cls) -> str:
        return TracyWrenchTopic.RIGHT_RAW

    @classproperty
    def median_readings(cls) -> int:
        """
        Wide enough for the spikes this sensor's data path produces: measured at up to
        12 N against a 0.9 N reading, isolated, and never less than three readings apart
        at the 100 Hz the driver publishes, so a window of five always holds a majority of
        good readings.
        """
        return 5

    @classproperty
    def gravity_frame(cls) -> str:
        return TracyFrame.LEVEL

    @classproperty
    def retare_service(cls) -> str:
        return TracyWrenchService.RIGHT

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "right_tool0")
        )


@dataclass(eq=False)
class TracyLeftArm(Arm[TracyLeftGripper], HasSensors[TracyLeftForceTorqueSensor]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(zip(connections, [2.62, -1.035, 1.13, -0.966, -0.88, 2.07])),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "table"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_wrist_3_link"
            ),
        )


@dataclass(eq=False)
class TracyRightArm(Arm[TracyRightGripper], HasSensors[TracyRightForceTorqueSensor]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(zip(connections, [3.72, -2.07, -1.17, 4.0, 0.82, 0.75])),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "table"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_wrist_3_link"
            ),
        )


@dataclass(eq=False)
class TracyCamera(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
            default_camera=True,
        )


@dataclass(eq=False)
class Tracy(
    AbstractRobot, HasLeftRightArm[TracyLeftArm, TracyRightArm], HasSensors[TracyCamera]
):
    """
    The dual UR10 arm setup used in the TraceBot project.

    https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_tracy_description/urdf/tracy.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "table"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "tracy.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        self.tighten_dof_velocity_limits_proportionally(maximum_velocity=0.2)

    def get_end_effectors(self) -> list[EndEffector]:
        return [self.left_arm.end_effector, self.right_arm.end_effector]
