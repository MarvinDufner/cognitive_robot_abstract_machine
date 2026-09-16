from __future__ import annotations

import inspect
import logging
import types
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from typing import (
    Optional,
    Self,
    TYPE_CHECKING,
    Set,
    List,
    DefaultDict,
    Type,
    Union,
    Any,
    cast,
)
from uuid import UUID

import numpy as np
from typing_extensions import get_origin, get_args, Generic, TypeVar, Unpack

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
)
from krrood.entity_query_language.factories import variable, contains, a, entity
from krrood.ormatic.utils import classproperty
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import VariableParameters
from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import (
    NoJointStateWithType,
    UselessConceptError,
    DuplicateRobotAssignmentsError,
    MissingDefaultCameraError,
    NoForceTorqueSensorForFrameError,
    NoForceTorqueSensorForTipError,
)
from semantic_digital_twin.robots.robot_part_mixins import (
    HasEndEffector,
    HasMobileBase,
    HasSensors,
    TGenericEndEffector,
    HasLeftRightArm,
    TGenericSensors,
    RobotPartMixin,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types import (
    Quaternion,
    Vector3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap, Derivatives
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
    WheeledDrive,
    ActiveConnection1DOF,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
    DegreeOfFreedom,
)
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    Connection,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.api import (
        BodySpecification,
        ConnectionSpecification,
    )
else:
    World = Any

logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class HasRobotParts(ABC):
    """
    Mixin for semantic annotations that have robot parts assigned to them.

    Provides methods for robot part aggregation, as well as handling the automatic setup
    of robot parts.
    """

    @property
    def _robot_parts(self) -> list[AbstractRobotPart]:
        """
        Serves as a generic interface to access all robot parts assigned to a robot
        part.

        Returns a list of all robot parts assigned directly to this robot part.
        """
        return self._aggregate_robot_parts(set())

    def _aggregate_robot_parts(self, seen: Set[UUID]) -> list[AbstractRobotPart]:
        """
        Recursively aggregates all robot parts assigned to this robot part, including
        itself if it is a robot part.

        Uses a set of seen UUIDs to avoid infinite recursion in case of cyclic
        references and duplicates.
        """
        introspector = DataclassOnlyIntrospector()
        robot_parts = []

        if isinstance(self, AbstractRobotPart):
            if self.id in seen:
                return []
            seen.add(self.id)
            robot_parts.append(self)

        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)

            if isinstance(value, list_like_classes):
                for robot_part in value:
                    if not isinstance(robot_part, HasRobotParts):
                        continue
                    robot_parts.extend(robot_part._aggregate_robot_parts(seen))
            elif isinstance(value, HasRobotParts):
                robot_parts.extend(value._aggregate_robot_parts(seen))

        return robot_parts

    def setup_robot_part_semantic_annotations(self):
        """
        Automatically discovers and initializes sub-parts by introspecting dataclass
        fields.
        """
        introspector = DataclassOnlyIntrospector()

        for attr in introspector.discover(self.__class__):
            field_name = attr.public_name
            field_type = attr.field.type

            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin in list_like_classes and args:
                self._initialize_list_field(field_name, args)

            elif (
                inspect.isclass(field_type)
                and issubclass(field_type, AbstractRobotPart)
                and not inspect.isabstract(field_type)
                and getattr(self, field_name) is None
            ):
                # Each robot part is only initialized once
                if any(type(item) is field_type for item in self._robot_parts):
                    continue

                part = field_type.setup_default_configuration_in_world_below_robot_root(
                    self.root
                )
                setattr(self, field_name, part)
                # Recursive call to trigger initialization for child part's fields
                part.setup_robot_part_semantic_annotations()

    def _initialize_list_field(self, field_name: str, types_to_initialize: list[Any]):
        """
        Helper to initialize all parts matching item_type and append them to a list
        field.

        :param field_name: Name of the list field to initialize
        :param types_to_initialize: List of types to initialize in the field
        """
        current_list = getattr(self, field_name)
        if not isinstance(current_list, list):
            current_list = []
            setattr(self, field_name, current_list)

        for concrete_type in types_to_initialize:
            if get_origin(concrete_type) in [Union, types.UnionType]:
                self._initialize_list_field(field_name, get_args(concrete_type))
                continue

            if (
                inspect.isclass(concrete_type)
                and issubclass(concrete_type, AbstractRobotPart)
                and not inspect.isabstract(concrete_type)
            ):
                if any(type(item) is concrete_type for item in self._robot_parts):
                    continue

                part = (
                    concrete_type.setup_default_configuration_in_world_below_robot_root(
                        self.root
                    )
                )
                current_list.append(part)
                # Recursive call for nested robot parts
                part.setup_robot_part_semantic_annotations()


@dataclass(eq=False)
class AbstractRobotPart(HasRootBody, HasRobotParts, ABC):
    """
    Abstract base class for all robot parts.

    A robot part is a part of a robot that can have its own kinematic structure and
    hardware interfaces, such as arms, sensors, or the mobile base. The robot property
    is computed lazily to avoid circular dependencies.
    """

    joint_states: list[JointState] = field(default_factory=list)
    """
    Common joint states for the current robot part.
    """

    @classmethod
    @abstractmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up a default configuration of this robot part in the world, below the given
        robot root.

        This is used to set up a default configuration of the robot part in the world
        after parsing a URDF.
        """

    @abstractmethod
    def setup_hardware_interfaces(self):
        """
        Sets up a default hardware interface for this robot part by setting the
        has_hardware_interface flag to True for relevant connections of this robot part.

        Implement as "pass" if this robot part does not have any hardware interfaces.
        """

    @abstractmethod
    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up default joint states for this robot part.

        Implement as "return []" if this robot part does not have any important joint
        states.
        """

    @synchronized_attribute_modification
    def add_joint_state(self, joint_state: JointState):
        """
        Adds a joint state to this semantic annotation.
        """
        self.joint_states.append(joint_state)
        joint_state.assign_to_robot(self._robot)

    def add_joint_states(self, joint_states: list[JointState]):
        """
        Adds multiple joint states to this semantic annotation.
        """
        for joint_state in joint_states:
            self.add_joint_state(joint_state)

    def get_joint_state_by_type(self, state_type: JointStateType) -> JointState:
        """
        Returns a JointState for a given joint state type.

        :param state_type: The state type to search for
        :return: The joint state with the given type
        """
        for j in self.joint_states:
            if j.state_type == state_type:
                return j
        raise NoJointStateWithType(state_type)

    def has_joint_state_of_type(self, state_type: JointStateType) -> bool:
        """
        Whether this part can be commanded into the given joint state.

        :param state_type: The state type to search for
        :return: True if a joint state of that type is defined
        """
        return any(
            joint_state.state_type == state_type for joint_state in self.joint_states
        )

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: str,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        parent_connection_specification: Optional[ConnectionSpecification] = None,
        scale: Optional[Scale] = None,
    ) -> Self:
        """
        Robot-part bodies originate from the parsed URDF, so they cannot be spawned from
        scratch.

        :raises UselessConceptError: Always, since robot-part bodies must already exist
            in the world.
        """
        raise UselessConceptError(
            reason="The bodies needed for RobotParts should already exist in the world after parsing a URDF"
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: Optional[str] = None,
        scale: Optional[Scale] = None,
        connection_specification: Optional[ConnectionSpecification] = None,
    ) -> BodySpecification:
        """
        Robot-part geometry comes from the parsed URDF, not from a scale, so a default
        body specification cannot be derived.

        :raises UselessConceptError: Always, since robot-part bodies must already exist
            in the world.
        """
        raise UselessConceptError(
            reason="The bodies needed for RobotParts should already exist in the world after parsing a URDF"
        )

    @property
    def _robot(self) -> Optional[AbstractRobot]:
        """
        Computes backreference to the robot this robot part belongs to.
        """
        robot_variable = variable(AbstractRobot, self._world.semantic_annotations)
        robot = (
            a(entity(robot_variable))
            .where(contains(robot_variable._robot_parts, self))
            .tolist()
        )
        if len(robot) == 0:
            return None
        elif len(robot) > 1:
            raise DuplicateRobotAssignmentsError(robot_part=self, robots=robot)
        return robot[0]

    def _setup_hardware_interfaces_for_active_connections(self):
        """
        Sets up a default hardware interface for the robot part by setting the
        has_hardware_interface flag to True for all active connections of all robot
        parts in this robot part.
        """
        for robot_part in self._robot_parts:
            for connection in robot_part.active_connections:
                connection.has_hardware_interface = True

    @property
    def active_connections(self) -> list[ActiveConnection]:
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection)
        ]


@dataclass(eq=False)
class KinematicChain(AbstractRobotPart, ABC):
    """
    A kinematic chain is a robot part that consists of a chain of bodies and connections
    between them.

    It has a root body and a tip body, and the connections between them can be computed
    using the world description.
    """

    tip: Body = field(kw_only=True)
    """
    The body at the end of the kinematic chain.
    """

    def _kinematic_structure_entities(
        self, visited: Set[int]
    ) -> list[KinematicStructureEntity]:
        """
        Computes the kinematic structure entities of this kinematic chain, which are the
        bodies and connections that make up the kinematic chain, including the bodies of
        any robot parts that are part of this kinematic chain.
        """
        if id(self) in visited:
            return []
        visited.add(id(self))
        kinematic_structure_entities = [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
        ]

        for robot_part in self._robot_parts:
            kinematic_structure_entities.extend(
                robot_part._kinematic_structure_entities(visited=visited)
            )

        return kinematic_structure_entities

    @property
    def connections(self) -> list[Connection]:
        """
        Returns the connections of the kinematic chain.

        This is a list of connections between the bodies in the kinematic chain
        """
        if self.root == self.tip:
            return []
        return self._world.compute_chain_of_connections(self.root, self.tip)

    def approximate_length(self) -> float:
        """
        Approximates the length of the kinematic chain by adding up  the distance
        between each body pair along the chain. For Prismatic Connections the upper
        limit of the connection is used, so the function returns the maximum length.

        :return: the approximate length of the kinematic chain
        """
        length = 0
        for connection in self.connections:
            parent_pose = connection.parent.global_pose
            child_pose = connection.child.global_pose
            dist = (
                connection.dof.limits.upper.position
                if isinstance(connection, PrismaticConnection)
                else parent_pose.to_position().euclidean_distance(
                    child_pose.to_position()
                )
            )
            length += dist
        return length


@dataclass(eq=False)
class Sensor(AbstractRobotPart, ABC):
    """
    Abstract base class for all sensors.

    A sensor is a robot part that can perceive the environment.
    """


@dataclass(eq=False)
class Camera(Sensor, ABC):
    """
    A camera is a sensor that captures images of the environment.
    """

    forward_facing_axis: Vector3 = field(kw_only=True)
    """
    The axis of the camera that is facing forward, expressed in the camera's root frame.
    """

    field_of_view: FieldOfView = field(kw_only=True)
    """
    The field of view of the camera, defined by the vertical and horizontal angles of
    the camera's view.
    """

    default_camera: bool = False
    """
    Whether this camera is the default camera of the robot.

    Used for quick access.
    """

    minimal_height: float = 0.0
    """
    The minimal height of the camera above the ground, in meters.
    """

    maximal_height: float = 1.0
    """
    The maximal height of the camera above the ground, in meters.
    """

    def __post_init__(self):
        super().__post_init__()
        self.forward_facing_axis.reference_frame = self.root

    @property
    def root_T_forward_view(self) -> HomogeneousTransformationMatrix:
        """
        The camera's pose in the world root frame, with its x axis along the direction
        the camera looks.

        The y and z axes only complete the frame and carry no meaning.
        """
        root_T_camera = self.root.global_transform
        root_V_forward = root_T_camera.to_rotation_matrix() @ self.forward_facing_axis
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_T_camera.to_position(),
            rotation_matrix=RotationMatrix.from_x_axis(root_V_forward),
            reference_frame=root_T_camera.reference_frame,
        )


class WrenchTopic(StrEnum):
    """
    The topics a force/torque reading travels on between the sensor and its consumers.
    """

    RAW = "/wrist_wrench/raw"
    """What the sensor itself publishes, still carrying the weight of the tool."""

    COMPENSATED = "/wrist_wrench/compensated"
    """The external contact wrench, once the load and the bias have been removed."""


@dataclass(frozen=True)
class ForceTorqueSensorLoad:
    """
    The static load a force/torque sensor carries, in the sensor frame.

    Describes what is mounted past the sensor, so a reading at rest can be predicted and
    subtracted, leaving only the external contact wrench:
    ``force_raw = force_offset + mass * gravity_in_sensor`` and
    ``torque_raw = torque_offset + first_moment x gravity_in_sensor``.
    """

    mass: float = 0.0
    """Mass mounted past the sensor, in kg."""

    first_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """First mass moment ``mass * centre_of_mass``, in kg m."""

    force_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Constant force bias of the unloaded sensor, in N."""

    torque_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Constant torque bias of the unloaded sensor, in N m."""


@dataclass(eq=False)
class ForceTorqueSensor(Sensor, ABC):
    """
    A six-axis force/torque sensor, measuring the wrench acting at its root frame.

    The sensor owns the symbolic force and torque of its live wrench together with the
    values behind them, mirroring how a degree of freedom owns its position symbol.
    Consumers reach the symbols through the robot annotation; producers push new
    readings through :meth:`write_wrench`.
    """

    force: Vector3 = field(init=False, default=None, compare=False, repr=False)
    """
    Symbolic force in the sensor frame, its live value held in :attr:`wrench_data`.
    """

    torque: Vector3 = field(init=False, default=None, compare=False, repr=False)
    """
    Symbolic torque in the sensor frame, its live value held in :attr:`wrench_data`.
    """

    wrench_data: FloatVariableData = field(
        init=False, default_factory=FloatVariableData, compare=False, repr=False
    )
    """
    Values behind :attr:`force` and :attr:`torque`. Sensor-owned and therefore of fixed
    size, so an expression compiled against it stays independent of how many nodes the
    motion statechart holds.
    """

    has_received_wrench: bool = field(
        init=False, default=False, compare=False, repr=False
    )
    """
    Whether a reading has been written, so consumers can tell an idle sensor from one
    measuring zero.
    """

    def __post_init__(self):
        super().__post_init__()
        self.force = Vector3.create_with_variables(f"{self.name}/force")
        self.force.reference_frame = self.root
        self.torque = Vector3.create_with_variables(f"{self.name}/torque")
        self.torque.reference_frame = self.root
        self.wrench_data.register_expression(self.force)
        self.wrench_data.register_expression(self.torque)

    def write_wrench(self, force: np.ndarray, torque: np.ndarray) -> None:
        """
        Overwrite the live wrench. The single entry point for every producer: a
        real-robot subscriber, a simulation, or a test.

        .. note:: Writes and returns; this never notifies the world of a state change.

        :param force: Measured force in the sensor frame.
        :param torque: Measured torque in the sensor frame.
        """
        self.wrench_data.set_value(self.force, force)
        self.wrench_data.set_value(self.torque, torque)
        self.has_received_wrench = True

    @classmethod
    def with_root(
        cls, world: World, frame: KinematicStructureEntity
    ) -> ForceTorqueSensor:
        """
        Return the sensor in ``world`` rooted at ``frame``.

        Resolving by frame rather than holding the annotation keeps consumers correct
        across the world's JSON round trip: the frame survives serialization, and the
        live annotation is looked up from the world that is actually executed.

        :param world: World holding the sensor annotation.
        :param frame: Frame the wanted sensor is rooted at.
        :raises NoForceTorqueSensorForFrameError: If no sensor is rooted at the frame.
        """
        for sensor in world.get_semantic_annotations_by_type(cls):
            if sensor.root == frame:
                return sensor
        raise NoForceTorqueSensorForFrameError(frame=frame)

    @classmethod
    def for_tip(cls, world: World, tip: KinematicStructureEntity) -> ForceTorqueSensor:
        """
        Return the sensor whose measurements pertain to ``tip``: the one rooted at a
        kinematic ancestor of ``tip``, closest to it when several lie on the chain.

        Resolving by the controlled tip keeps a caller agnostic of the robot's wiring
        and disambiguates multi-arm robots, where each arm's tip selects its own sensor.

        :param world: World holding the sensor annotation.
        :param tip: Tip whose measurements are wanted.
        :raises NoForceTorqueSensorForTipError: If no sensor lies on the chain.
        """
        chain = world.compute_chain_of_kinematic_structure_entities(world.root, tip)
        depth_by_entity = {entity: depth for depth, entity in enumerate(chain)}

        def depth_on_chain(sensor: ForceTorqueSensor) -> Optional[int]:
            """
            How far along the chain the sensor sits, following the bodies it is bolted
            to. A sensor frame is often a leaf beside the chain rather than a link of it.
            """
            entity = sensor.root
            while entity is not None:
                if entity in depth_by_entity:
                    return depth_by_entity[entity]
                connection = entity.parent_connection
                if not isinstance(connection, FixedConnection):
                    return None
                entity = connection.parent
            return None

        depths = {
            sensor: depth_on_chain(sensor)
            for sensor in world.get_semantic_annotations_by_type(cls)
        }
        on_chain = {
            sensor: depth for sensor, depth in depths.items() if depth is not None
        }
        if not on_chain:
            raise NoForceTorqueSensorForTipError(tip=tip)
        return max(on_chain, key=on_chain.get)

    @classproperty
    def wrench_topic(cls) -> Optional[str]:
        """
        The topic this kind of sensor's readings arrive on.

        ``None`` for a sensor that is not fed from ROS, which is every sensor written
        directly by a simulation or a test.
        """
        return None

    @property
    def measured_force(self) -> np.ndarray:
        """
        The force of the last reading, in the sensor frame, in N.

        Reads the values behind :attr:`force`, which is registered first, rather than
        evaluating the expression, which a control cycle cannot afford.
        """
        return self.wrench_data.data[:3]

    @classproperty
    def median_readings(cls) -> int:
        """
        How many of the most recent readings the live wrench is the median of.

        One writes every reading through. More drops the isolated spikes a sensor's data
        path can produce, at the cost of half the window in delay, and needs a window
        wider than the longest run of spikes it has to survive.
        """
        return 1

    @classproperty
    def raw_wrench_topic(cls) -> Optional[str]:
        """
        The topic this kind of sensor's driver publishes, still carrying the load.

        What a compensation node reads; ``None`` for a sensor nothing compensates.
        """
        return None

    @classproperty
    def gravity_frame(cls) -> Optional[str]:
        """
        A frame of this robot whose z axis points up.

        Removing the load's weight needs the sensor's orientation against gravity, and
        only the robot knows which of its frames is level.
        """
        return None

    @classproperty
    def retare_service(cls) -> Optional[str]:
        """
        The service that zeroes this kind of sensor's compensation.

        ``None`` for a sensor whose readings need no zeroing, or that nothing
        compensates.
        """
        return None

    @classproperty
    def load(cls) -> ForceTorqueSensorLoad:
        """
        The static load this kind of sensor carries.

        A property of the sensor model rather than of one instance, so a tool that has no
        world, such as an offline bag check, can still ask for it. Defaults to nothing
        mounted, so an uncalibrated sensor compensates to its own raw reading.
        """
        return ForceTorqueSensorLoad()


@dataclass(eq=False)
class Finger(KinematicChain, ABC):
    """
    A finger is a kinematic chain attached to a gripper to manipulate objects.
    """

    finger_tip_frame: Optional[Body] = None
    """
    The frame of the finger tip.

    Could be used to align the finger with, for example, a button.
    """


@dataclass(eq=False)
class EndEffector(AbstractRobotPart, ABC):
    """
    Abstract base class of robot end effector.

    Always has a tool frame.
    """

    tool_frame: Body = field(kw_only=True)
    """
    The tool frame or tool center point of the end_effector.

    Usually the point the robot tries to align with the object.
    """

    front_facing_orientation: Quaternion = field(kw_only=True)
    """
    The orientation of the end_effector's tool frame, which is usually the front-facing
    orientation.
    """

    front_facing_axis: Vector3 = field(init=False)
    """
    The axis of the end_effector's tool frame that is facing forward.
    """

    def __post_init__(self):
        super().__post_init__()
        rotation_matrix = RotationMatrix.from_quaternion(self.front_facing_orientation)
        self.front_facing_axis = Vector3.from_iterable(rotation_matrix[:3, 0])


@dataclass(eq=False)
class Torso(KinematicChain, ABC):
    """
    The torso of a robot, which is a kinematic chain providing additional shared degrees
    of freedom to its attachments, such as arms or the neck.
    """


@dataclass(eq=False)
class Arm(KinematicChain, HasEndEffector[TGenericEndEffector], ABC):
    """
    An arm is a kinematic chain that has an end effector attached to it.
    """

    def maximum_reach(self, samples_per_degree_of_freedom: int = 5) -> float:
        """
        The largest horizontal distance between the robot's root and this arm's tool
        frame, in m.

        A purely kinematic upper bound: no inverse kinematics and no collision checking,
        so a caller gets an optimistic reach that geometric planning can work with.

        .. note:: Unlike :meth:`KinematicChain.approximate_length`, this measures how far
            the tool can actually be placed horizontally rather than the length of the
            chain, so a folded or vertically stacked arm is not overestimated.

        :param samples_per_degree_of_freedom: How many positions each degree of freedom
            is sampled at. Interior samples are required because the farthest reach sits
            at an intermediate joint angle rather than at a limit.
        """
        # Both bounds are required: has_position_limits already holds when only one
        # side is set, which is not an interval that can be sampled.
        degrees_of_freedom = [
            dof
            for connection in self.connections
            if isinstance(connection, ActiveConnection)
            for dof in connection.active_dofs
            if dof.has_position_limits
            and dof.limits.lower.position is not None
            and dof.limits.upper.position is not None
        ]
        root_P_tool = self._world.compose_forward_kinematics_expression(
            self._robot.root, self.end_effector.tool_frame
        ).to_position()
        horizontal_distance = (
            Vector3(x=root_P_tool.x, y=root_P_tool.y)
            .norm()
            .compile(
                parameters=VariableParameters.from_lists(
                    self._world.state.position_float_variables
                )
            )
        )
        samples = [
            np.linspace(
                dof.limits.lower.position,
                dof.limits.upper.position,
                samples_per_degree_of_freedom,
            )
            for dof in degrees_of_freedom
        ]
        initial_positions = self._world.state.positions.copy()
        reach = 0.0
        for configuration in product(*samples):
            for dof, position in zip(degrees_of_freedom, configuration):
                self._world.state[dof.id].position = position
            reach = max(reach, horizontal_distance(self._world.state.positions).item())
        self._world.state.set_derivative(Derivatives.position, initial_positions)
        self._world.notify_state_change()
        return reach


@dataclass(eq=False)
class Neck(
    KinematicChain,
    HasSensors[Unpack[TGenericSensors]],
    ABC,
):
    """
    The neck of a robot, which is a kinematic chain that has a camera attached to it.
    """


TGenericDrive = TypeVar("TGenericDrive", bound=WheeledDrive)


@dataclass(eq=False)
class MobileBase(AbstractRobotPart, Generic[TGenericDrive], ABC):
    """
    The base of a robot.

    The drive connection attaching the base to its ``odom`` frame is bound as the
    generic parameter (e.g. ``MobileBase[OmniDrive]``) by each concrete mobile base.
    """

    full_body_controlled: bool = field(default=False, kw_only=True)
    """
    If True, the robot can move its entire body during a motion.

    If False, only the robot will always stand still when moving an arm.
    """

    @classproperty
    @abstractmethod
    def forward_axis(cls) -> Vector3:
        """
        The axis of this base that points where the robot faces.
        """

    def pose_facing(self, heading: Pose) -> Pose:
        """
        The base pose whose :attr:`forward_axis` points along ``heading``.

        ``heading``'s orientation says where the robot's front should point, written as
        its x-axis, so the same heading serves bases modelled with different axes. Its
        position is kept as it is.
        """
        base_R_forward = RotationMatrix.from_vectors(x=self.forward_axis, z=Vector3.Z())
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            heading.to_position(),
            heading.to_rotation_matrix() @ base_R_forward.inverse(),
            reference_frame=heading.reference_frame,
        ).to_pose()

    @classmethod
    def get_drive_connection_type(cls) -> Type[TGenericDrive]:
        """
        The connection type attaching this mobile base to its ``odom`` frame.

        Resolved from the generic drive parameter bound by the concrete mobile base.
        """
        return get_generic_type_parameters(cls, MobileBase)[0]

    @property
    def bounding_box(self) -> VolumetricBoundingBox:
        return self.root.collision.as_bounding_box_collection_in_frame(
            self._world.root
        ).bounding_box()

    @property
    def footprint_radius(self) -> float:
        """
        Horizontal radius of the base, in m: how far it extends from its own origin, and
        therefore how close that origin can be placed to an obstacle.
        """
        box = self.bounding_box
        return max(box.depth, box.width) / 2.0


@dataclass(eq=False)
class AbstractRobot(Agent, HasRobotParts, ABC):
    """
    This implementation was initially introduced in https://github.com/cram2/cognitive_robot_abstract_machine/pull/290
    To see a more detailed account of the reasoning, refer to that PRs description.

    ---------------------------------------------------------------------------------------------

    Specification of an abstract robot and its semantic annotations.

    This class serves as the foundation for robot handling within the framework,
    designed to ensure consistency and expressiveness in robot definitions.

    Design Evolution and Rationale
    ------------------------------
    Before settling on the current architecture, two primary approaches were
    considered for representing diverse robot structures:

    1.  **Unified Base Class**: Providing every robot part with all possible
        fields (e.g., arms, torso, mobile base) regardless of actual hardware.
        This was rejected because it led to redundant data, confusing APIs where
        most fields remained ``None``, and a significant maintenance burden to
        keep duplicated information synchronized.
    2.  **Specialized Structures (Chosen)**: Defining only the fields relevant
        to a specific robot part (e.g., ``Tracy.arms``, but
        ``PR2.mobile_base.torso.arms``). This approach was chosen because it
        accurately describes any robot structure without duplication.

    To overcome the lack of deep type-hinting in specialized structures, the
    framework utilizes Python Generics and the ``SubclassSafeGeneric`` pattern.
    While this involves more "under-the-hood" complexity, it was judged
    superior to alternatives like ``typing.Annotated`` or manual field
    overrides, which require excessive boilerplate and increase the risk of
    developer error.

    Implementation and Automation
    -----------------------------
    To reduce the learning curve and prevent invalid world states, several
    critical processes are automated:

    *   **Synchronization**: The framework automatically handles the order in
        which semantic annotations are added to the world, removing the need
        for developers to understand complex internal synchronization
        requirements.
    *   **Initialization**: Sub-parts are automatically instantiated based on
        generic type hints, ensuring that robot structures are valid by
        construction.

    Rules for Implementing a New Robot
    ----------------------------------
    When implementing a new robot, follow these three rules:

    1.  **Map Concepts**: Create a new class for every distinct part of the
        robot defined in ``robot_parts.py``.
    2.  **Define Hierarchy**: Use mixins and generics to define direct
        parent-child relationships (e.g.,
        ``PR2RightArm(HasEndEffector[PR2RightGripper])``).
    3.  **Implement Abstract Methods**: Fill in the required abstract methods.
        If a method does not apply (e.g., no hardware interface), a simple
        ``pass`` is sufficient.

    Validation
    ----------
    Call the ``validate()`` method to confirm that all fields are plausibly
    filled and that the robot can be synchronized without issues.
    """

    @classmethod
    @abstractmethod
    def get_ros_file_path(cls) -> str:
        """
        Returns a ROS file path pointing to the description of this robot, for example a
        URDF file.
        """

    @classmethod
    @abstractmethod
    def _get_root_body_name(cls) -> str:
        """
        Returns the name of the root body of the robot in the world, which serves as the
        entry point for traversing the robot's kinematic structure.
        """

    @classmethod
    def get_drive_connection_type(cls) -> Type[Connection]:
        """
        The connection type attaching this robot to its ``odom`` frame.

        :return: The mobile base's drive connection type, or :class:`FixedConnection`
            when the robot has no mobile base and is therefore rigidly attached.
        """
        if not issubclass(cls, HasMobileBase):
            return FixedConnection
        mobile_base_type = cast(
            MobileBase, get_generic_type_parameters(cls, HasMobileBase)[0]
        )
        return mobile_base_type.get_drive_connection_type()

    def setup_robot_part_semantic_annotations(self):
        """
        Sets up the semantic annotations for all robot parts of this robot.
        """
        super().setup_robot_part_semantic_annotations()

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a robot from a world.
        """
        return cls.from_branch_in_world(world.root)

    @classmethod
    def from_branch_in_world(cls, branch_root: KinematicStructureEntity) -> Self:
        """
        Creates a robot from a branch in a world.

        This is useful when you have multiple of the same robots in the same world,
        which would normally cause naming conflicts.
        """
        world = branch_root._world
        robot_root = world.get_body_in_branch_by_name(
            branch_root=branch_root, name=cls._get_root_body_name()
        )
        with world.modify_world():
            self = cls(
                root=robot_root,
            )
            self.setup_robot_part_semantic_annotations()
            world.add_semantic_annotation_recursively(self)
            for robot_part in self._robot_parts:
                robot_part.setup_hardware_interfaces()
                robot_part.add_joint_states(robot_part.setup_joint_states())
            self._setup_collision_rules()
            self._setup_velocity_limits()
            return self

    @property
    def controlled_connections(self) -> list[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection) and connection.is_controlled
        ]

    @property
    def degrees_of_freedom_with_hardware_interface(self) -> List[DegreeOfFreedom]:
        """
        The number of degrees of freedom of the robot, which is the sum of the degrees
        of freedom of all its end_effectors.
        """
        dofs_with_hardware_interfaces = []
        for connection in self.connections:
            dofs = connection.controlled_dofs
            for dof in dofs:
                if dof in dofs_with_hardware_interfaces:
                    continue
                dofs_with_hardware_interfaces.append(dof)
        return dofs_with_hardware_interfaces

    def validate(self) -> bool:
        """
        Validates the robot semantic annotation.
            The validation process includes:
            1. Deepcopy the resulting world to ensure that all parts of the robot are initialized in the correct order
            2. Assert that the copied world is the same as the original world
            3. Assert that the robot semantic annotation has a default camera.
            4. Call validate method on all robot parts inheriting froma RobotPartMixin

        :return: True if the robot semantic annotation is valid, False otherwise.
        """
        self_world_copy = deepcopy(self._world)

        assert set(self_world_copy._world_entity_hash_table.keys()) == set(
            self._world._world_entity_hash_table.keys()
        )

        assert (
            self_world_copy.get_semantic_annotations_by_type(AbstractRobot)[
                0
            ].get_default_camera()
            is not None
        )

        for part in self._robot_parts:
            assert part._robot == self, f"Part {part} refers to wrong robot"

            if isinstance(part, RobotPartMixin):
                part.validate()

        return True

    def _setup_velocity_limits(self):
        """
        Sets up velocity limits for 1-DOF connections in the robot.
        """
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_proportionally(maximum_velocity=1)

    @property
    def drive(self) -> Optional[WheeledDrive]:
        """
        The connection which the robot uses for driving.
        """
        try:
            parent_connection = self.root.parent_connection
            if isinstance(parent_connection, WheeledDrive):
                return parent_connection
        except AttributeError:
            pass

    def set_root_pose(self, pose: Pose) -> None:
        """
        Place the robot's root at ``pose``.

        A pose that is not already expressed in the root connection's parent frame is
        converted into it, so the robot lands at ``pose`` no matter how many frames (an
        ``odom``, for example) sit between that frame and the pose's own.

        ..note:: A drive that cannot represent every degree of freedom applies only what
            it can, so the root reaches ``pose`` only within the drive's own limits.

        :param pose: The pose the robot's root should end up at.
        """
        connection = self.root.parent_connection
        parent_kinematic_structure_entity = connection.parent
        if pose.reference_frame is not parent_kinematic_structure_entity:
            pose = self._world.transform(pose, parent_kinematic_structure_entity)

        connection.origin = pose.to_homogeneous_matrix()

    @property
    def _one_dof_connections(self) -> list[ActiveConnection1DOF]:
        """
        All 1-DOF active connections that belong to this robot.

        Velocity limit adjustments must only touch the robot's own joints, never
        unrelated environment joints (drawers, doors, ...) in the same world.
        """
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def tighten_dof_velocity_limits_of_1dof_connections(
        self,
        new_limits: DefaultDict[ActiveConnection1DOF, float],
    ):
        """
        Convenience method for tightening the velocity limits of all one degree-of-
        freedom (1DOF) active connections in the system.

        The method iterates through all connections of type `ActiveConnection1DOF` and
        configures their velocity limits by overwriting the existing lower and upper
        limit values with the provided ones.

        :param new_limits: A dictionary linking 1DOF connections to their corresponding
            new velocity limits. The keys are of type `ActiveConnection1DOF`, and the
            values represent the new velocity limits specific to each connection.
        """
        for connection in self._one_dof_connections:
            connection.raw_dof._overwrite_dof_limits(
                new_lower_limits=DerivativeMap(
                    None, -new_limits[connection], None, None
                ),
                new_upper_limits=DerivativeMap(
                    None, new_limits[connection], None, None
                ),
            )

    def tighten_dof_velocity_limits_proportionally(
        self, maximum_velocity: float
    ) -> None:
        """
        Tightens the velocity limits of all 1-DOF active connections proportionally,
        preserving the relative magnitudes defined in the original robot description.

        The joint with the highest current velocity limit is mapped to
        ``maximum_velocity``; all others are scaled by the same factor.
        Joints with no velocity limit are left unchanged.

        If the current maximum is already at or below ``maximum_velocity``,
        no changes are applied.

        :param maximum_velocity: The target velocity for the joint with the
            highest current velocity limit.
        """
        connections_with_velocity_limits = [
            (connection, connection.raw_dof.limits.upper.velocity)
            for connection in self._one_dof_connections
            if connection.raw_dof.limits.upper.velocity is not None
        ]
        if not connections_with_velocity_limits:
            return
        original_maximum = max(
            velocity for _, velocity in connections_with_velocity_limits
        )
        if original_maximum <= maximum_velocity:
            return
        scale_factor = maximum_velocity / original_maximum
        for connection, current_velocity in connections_with_velocity_limits:
            scaled_limit = current_velocity * scale_factor
            connection.raw_dof._overwrite_dof_limits(
                new_lower_limits=DerivativeMap(None, -scaled_limit, None, None),
                new_upper_limits=DerivativeMap(None, scaled_limit, None, None),
            )

    def get_end_effectors(self) -> list[EndEffector]:
        return [p for p in self._robot_parts if isinstance(p, EndEffector)]

    def get_arms(self) -> list[Arm]:
        return [p for p in self._robot_parts if isinstance(p, Arm)]

    def get_sensors(self) -> list[Sensor]:
        return [p for p in self._robot_parts if isinstance(p, Sensor)]

    def get_torso(self):
        [torso] = [p for p in self._robot_parts if isinstance(p, Torso)]
        return torso

    def get_torso_if_specified(self) -> Optional[Torso]:
        """
        :return: The robot's torso, or None for a robot built without one.
        """
        for part in self._robot_parts:
            if isinstance(part, Torso):
                return part
        return None

    def get_left_arm_if_specified(self) -> Optional[Arm]:
        if isinstance(self, HasLeftRightArm):
            return self.left_arm
        for part in self._robot_parts:
            if isinstance(part, HasLeftRightArm):
                return part.left_arm
        return None

    def get_right_arm_if_specified(self) -> Optional[Arm]:
        if isinstance(self, HasLeftRightArm):
            return self.right_arm
        for part in self._robot_parts:
            if isinstance(part, HasLeftRightArm):
                return part.right_arm
        return None

    def get_default_camera(self) -> Camera:
        """
        Returns the default camera of the robot.
        """
        for robot_part in self._robot_parts:
            if isinstance(robot_part, Camera) and robot_part.default_camera:
                return robot_part
        raise MissingDefaultCameraError(type(self))

    @abstractmethod
    def _setup_collision_rules(self):
        """
        Sets up collision rules for the robot.
        """
