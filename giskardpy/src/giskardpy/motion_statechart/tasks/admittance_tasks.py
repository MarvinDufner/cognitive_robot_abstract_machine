from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import NonPositiveVirtualMassError
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionTrajectory,
)
import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    VariableParameters,
    limit,
    vstack,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.spatial_types import Point3, Vector3

AXES = ("x", "y", "z")
"""
Names of the goal frame axes, in the order the admittance state is laid out.
"""


def yielding_step(
    measured: Vector3,
    desired: Vector3,
    mass: Vector3,
    damping: Vector3,
    stiffness: Vector3,
    maximum_offset: Vector3,
    control_dt: float,
    positions: List[FloatVariable],
    velocities: List[FloatVariable],
) -> sm.Vector:
    """
    One step of a diagonal virtual mass-damper-spring driven by how far the measurement is
    from what was asked for.

    Each axis integrates its own ``mass * acceleration + damping * velocity +
    stiffness * position = measured - desired``. The integration is semi-implicit Euler
    with the damping and stiffness terms taken implicitly, which is stable for any
    positive parameters; an explicit damping term instead diverges once
    ``control_dt * damping / mass`` exceeds 2.

    :return: The next offsets followed by the next rates, stacked.
    :raises NonPositiveVirtualMassError: If any axis has a non-positive mass.
    """
    next_positions, next_velocities = [], []
    for axis in range(len(AXES)):
        position, velocity = positions[axis], velocities[axis]
        error = measured[axis] - desired[axis]
        next_velocity = (
            mass[axis] * velocity + (error - stiffness[axis] * position) * control_dt
        ) / (
            mass[axis] + damping[axis] * control_dt + stiffness[axis] * control_dt**2
        )
        next_positions.append(
            limit(
                position + next_velocity * control_dt,
                -maximum_offset[axis],
                maximum_offset[axis],
            )
        )
        next_velocities.append(next_velocity)
    return vstack(next_positions + next_velocities)


@dataclass(eq=False, repr=False)
class AdmittanceCartesianTrajectory(CartesianPositionTrajectory):
    """
    Follow a trajectory while yielding to contact force.

    The tip is pulled towards the trajectory point offset by a virtual
    mass-damper-spring, driven by the difference between the measured and the desired
    contact force. The offset carries across the whole trajectory rather than being
    rebuilt per waypoint, so the press stays settled while the tool travels; resetting it
    at every waypoint would step the goal by however much compliance had accumulated.

    With no stiffness the offset also seeks contact on its own: away from the surface the
    force error is the full desired force, so the tool drifts towards it, up to
    :attr:`maximum_offset`, until it touches and the forces balance. The commanded surface
    height therefore need not be exact.

    .. todo:: Two refinements are unimplemented: post-sensor inertia compensation, which
        renders ``mass - inertia_compensation`` and belongs on the force/torque sensor
        annotation because it describes what is mounted past the sensor, and acceleration
        feedforward, which adds phase lead to the goal and is controller tuning rather
        than robot knowledge.
    """

    desired_force: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Contact force the admittance balances, in the goal frame.

    Biasing a few newtons into the surface removes tap-and-bounce.
    """

    mass: Vector3 = field(
        default_factory=lambda: Vector3(x=1.0, y=1.0, z=1.0), kw_only=True
    )
    """
    Virtual mass per axis, in kg.
    """

    damping: Vector3 = field(
        default_factory=lambda: Vector3(x=20.0, y=20.0, z=100.0), kw_only=True
    )
    """
    Virtual damping per axis, in N s/m.

    The surface normal is damped hardest so the press settles instead of bouncing, while
    the other axes yield sideways.
    """

    stiffness: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Virtual stiffness per axis, in N/m.

    Zero lets the offset persist once the force is balanced, instead of being pulled
    back to the nominal trajectory.
    """

    maximum_offset: Vector3 = field(
        default_factory=lambda: Vector3(x=0.02, y=0.02, z=0.02), kw_only=True
    )
    """
    How far the offset may take the tool from the trajectory, per axis, in m.

    It absorbs how far the surface really is from where the trajectory says it is, which
    is a matter of centimetres; without a bound the offset integrates the unbalanced
    desired force for as long as the arm takes to catch up, and arrives metres away.
    """

    _admittance_position: Optional[Vector3] = field(init=False, default=None)
    """
    Symbolic offset added to the tracked point, registered as a float variable so the QP
    reads the value written each tick.
    """

    _state: np.ndarray = field(init=False, default_factory=lambda: np.zeros(6))
    """
    Live offset and its rate: the input and the output of :attr:`_compiled_step`.
    """

    _compiled_step: Optional[CompiledFunction] = field(init=False, default=None)
    """
    Wrench rotation and one integration step, compiled together in
    :meth:`build_artifacts`.
    """

    _force_torque_sensor: Optional[ForceTorqueSensor] = field(init=False, default=None)
    """
    Live sensor annotation, resolved from :attr:`tip_link`.
    """

    @property
    def tracked_point(self) -> Point3:
        return self.goal_reference_frame_P_current_target_point + (
            self._admittance_position
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Compile the admittance before the trajectory constrains the tip to the offset
        point.

        :param context: Provides the world model and the control period.
        :return: The artifacts of this task.
        """
        self._force_torque_sensor = ForceTorqueSensor.for_tip(
            context.world, self.tip_link
        )
        self._compile_step(context)
        return super().build_artifacts(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self._force_torque_sensor.has_received_wrench:
            # Copies, because the compiled function hands back its own output buffer.
            self._state[:] = self._compiled_step(
                context.world.state.positions,
                self._force_torque_sensor.wrench_data.data,
                self._state,
            )
            context.float_variable_data.set_value(
                self._admittance_position, self._state[:3]
            )
        return super().on_tick(context)

    def on_reset(self, context: MotionStatechartContext) -> None:
        self._state[:] = 0.0
        context.float_variable_data.set_value(
            self._admittance_position, self._state[:3]
        )

    def _compile_step(self, context: MotionStatechartContext) -> None:
        """
        Compile the live wrench, rotated into the goal frame and integrated by one step
        of the virtual dynamics, into :attr:`_compiled_step`.

        The dynamics are diagonal, so each axis integrates its own
        ``mass * acceleration + damping * velocity + stiffness * position =
        measured_force - desired_force``. The integration is semi-implicit Euler with the
        damping and stiffness terms taken implicitly, which is stable for any positive
        parameters; an explicit damping term instead diverges once
        ``control_dt * damping / mass`` exceeds 2.

        The offset symbols are registered as float variables so the QP can read them,
        while the step takes them as its own parameter group. Together with the sensor's
        own wrench group, every group the step compiles against is of fixed size, so the
        step stays independent of how many nodes the chart holds.

        :param context: Provides the world model and the control period.
        :raises NonPositiveVirtualMassError: If any axis has a non-positive mass.
        """
        if any(float(self.mass[axis]) <= 0 for axis in range(len(AXES))):
            raise NonPositiveVirtualMassError(node=self, mass=self.mass)

        position_variables = [
            FloatVariable(name=f"{self.name}/admittance_position.{axis}")
            for axis in AXES
        ]
        velocity_variables = [
            FloatVariable(name=f"{self.name}/admittance_velocity.{axis}")
            for axis in AXES
        ]
        self._admittance_position = Vector3(
            *position_variables, reference_frame=self.goal_reference_frame
        )
        context.float_variable_data.register_expression(self._admittance_position)

        goal_T_sensor = context.world.compose_forward_kinematics_expression(
            self.goal_reference_frame, self._force_torque_sensor.root
        )
        measured_force = goal_T_sensor.rotation_matrix @ self._force_torque_sensor.force
        control_dt = context.qp_controller_config.control_time_step.total_seconds()

        self._compiled_step = yielding_step(
            measured=measured_force,
            desired=self.desired_force,
            mass=self.mass,
            damping=self.damping,
            stiffness=self.stiffness,
            maximum_offset=self.maximum_offset,
            control_dt=control_dt,
            positions=position_variables,
            velocities=velocity_variables,
        ).compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                self._force_torque_sensor.wrench_data.variables,
                position_variables + velocity_variables,
            ),
            sparse=False,
        )


# %% yielding in orientation


@dataclass(eq=False, repr=False)
class AdmittanceAlignPlanes(AlignPlanes):
    """
    Turn a tool's plane onto a surface, letting the measured torque say where the
    surface is.

    A tool lying flat on a surface feels no torque about it; one resting on an edge does,
    because the contact is then off to one side. The normal the tip is turned onto yields
    to that torque, so the normal the model supplies is only a starting guess and contact
    corrects it. That is what a surface whose pose is not known exactly needs, and it
    leaves the tool flat on a surface that is tilted or uneven.

    .. note:: Alone this only decides an orientation. It is meant beside a task that holds
        the tool against the surface, which is what produces the torque it reads.
    """

    desired_torque: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Contact torque the tool is turned to balance, in the goal normal's frame, in N m.

    Zero leaves it lying flat.
    """

    mass: Vector3 = field(
        default_factory=lambda: Vector3(x=0.1, y=0.1, z=0.1), kw_only=True
    )
    """
    Virtual moment of inertia per axis, in kg m^2.
    """

    damping: Vector3 = field(
        default_factory=lambda: Vector3(x=2.0, y=2.0, z=2.0), kw_only=True
    )
    """
    Virtual damping per axis, in N m s.

    Sets how fast the normal follows the torque.
    """

    stiffness: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Virtual stiffness per axis, in N m.

    Zero lets the correction persist once the torque is balanced, instead of being
    pulled back to the normal the model supplied.
    """

    maximum_offset: Vector3 = field(
        default_factory=lambda: Vector3(x=0.2, y=0.2, z=0.2), kw_only=True
    )
    """
    How far the correction may turn the normal, per axis, in rad.

    It absorbs how far the surface really is from the pose the model gives it, which is
    a matter of degrees; without a bound an unbalanced torque turns the normal without
    end.
    """

    _admittance_rotation: Optional[Vector3] = field(init=False, default=None)
    """
    Symbolic turn applied to the goal normal, registered as float variables so the QP
    reads the value written each tick.
    """

    _state: np.ndarray = field(init=False, default_factory=lambda: np.zeros(6))
    """
    Live correction and its rate: the input and the output of :attr:`_compiled_step`.
    """

    _compiled_step: Optional[CompiledFunction] = field(init=False, default=None)
    """
    Torque rotation and one integration step, compiled together.
    """

    _force_torque_sensor: Optional[ForceTorqueSensor] = field(init=False, default=None)
    """
    Live sensor annotation, resolved from :attr:`tip_link`.
    """

    @property
    def tracked_normal(self) -> Vector3:
        if self._admittance_rotation is None:
            return self.goal_normal
        # A small turn of a vector is the vector plus the turn crossed into it, which
        # stays smooth where an axis and an angle would divide by a vanishing angle.
        return self.goal_normal + self._admittance_rotation.cross(self.goal_normal)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Compile the correction before the alignment constrains the tip to the turned
        normal.

        :param context: Provides the world model and the control period.
        :return: The artifacts of this task.
        """
        self._force_torque_sensor = ForceTorqueSensor.for_tip(
            context.world, self.tip_link
        )
        self._compile_step(context)
        return super().build_artifacts(context)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self._force_torque_sensor.has_received_wrench:
            # Copies, because the compiled function hands back its own output buffer.
            self._state[:] = self._compiled_step(
                context.world.state.positions,
                self._force_torque_sensor.wrench_data.data,
                self._state,
            )
            context.float_variable_data.set_value(
                self._admittance_rotation, self._state[:3]
            )
        return super().on_tick(context)

    def on_reset(self, context: MotionStatechartContext) -> None:
        self._state[:] = 0.0
        context.float_variable_data.set_value(
            self._admittance_rotation, self._state[:3]
        )

    def _compile_step(self, context: MotionStatechartContext) -> None:
        """
        Compile the live torque, rotated into the goal normal's frame and integrated by
        one step of the virtual dynamics, into :attr:`_compiled_step`.

        :param context: Provides the world model and the control period.
        :raises NonPositiveVirtualMassError: If any axis has a non-positive inertia.
        """
        if any(float(self.mass[axis]) <= 0 for axis in range(len(AXES))):
            raise NonPositiveVirtualMassError(node=self, mass=self.mass)

        goal_frame = self.goal_normal.reference_frame
        position_variables = [
            FloatVariable(name=f"{self.name}/admittance_rotation.{axis}")
            for axis in AXES
        ]
        velocity_variables = [
            FloatVariable(name=f"{self.name}/admittance_angular_rate.{axis}")
            for axis in AXES
        ]
        self._admittance_rotation = Vector3(
            *position_variables, reference_frame=goal_frame
        )
        context.float_variable_data.register_expression(self._admittance_rotation)

        goal_T_sensor = context.world.compose_forward_kinematics_expression(
            goal_frame, self._force_torque_sensor.root
        )
        measured_torque = (
            goal_T_sensor.rotation_matrix @ self._force_torque_sensor.torque
        )
        self._compiled_step = yielding_step(
            measured=measured_torque,
            desired=self.desired_torque,
            mass=self.mass,
            damping=self.damping,
            stiffness=self.stiffness,
            maximum_offset=self.maximum_offset,
            control_dt=context.qp_controller_config.control_time_step.total_seconds(),
            positions=position_variables,
            velocities=velocity_variables,
        ).compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                self._force_torque_sensor.wrench_data.variables,
                position_variables + velocity_variables,
            ),
            sparse=False,
        )
