from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import NonPositiveVirtualMassError
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionTrajectory,
)
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    VariableParameters,
    vstack,
)
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.spatial_types import Point3, Vector3

AXES = ("x", "y", "z")
"""Names of the goal frame axes, in the order the admittance state is laid out."""


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
    force error is the full desired force, so the tool drifts towards it until it touches
    and the forces balance. The commanded surface height therefore need not be exact.

    .. todo:: Two refinements are unimplemented: post-sensor inertia compensation, which
        renders ``mass - inertia_compensation`` and belongs on the force/torque sensor
        annotation because it describes what is mounted past the sensor, and acceleration
        feedforward, which adds phase lead to the goal and is controller tuning rather
        than robot knowledge.
    """

    desired_force: Vector3 = field(default_factory=Vector3, kw_only=True)
    """Contact force the admittance balances, in the goal frame. Biasing a few newtons
    into the surface removes tap-and-bounce."""

    mass: Vector3 = field(
        default_factory=lambda: Vector3(x=1.0, y=1.0, z=1.0), kw_only=True
    )
    """Virtual mass per axis, in kg."""

    damping: Vector3 = field(
        default_factory=lambda: Vector3(x=20.0, y=20.0, z=100.0), kw_only=True
    )
    """Virtual damping per axis, in N s/m. The surface normal is damped hardest so the
    press settles instead of bouncing, while the other axes yield sideways."""

    stiffness: Vector3 = field(default_factory=Vector3, kw_only=True)
    """Virtual stiffness per axis, in N/m. Zero lets the offset persist once the force is
    balanced, instead of being pulled back to the nominal trajectory."""

    _admittance_position: Optional[Vector3] = field(init=False, default=None)
    """Symbolic offset added to the tracked point, registered as a float variable so the
    QP reads the value written each tick."""

    _state: np.ndarray = field(init=False, default_factory=lambda: np.zeros(6))
    """Live offset and its rate: the input and the output of :attr:`_compiled_step`."""

    _compiled_step: Optional[CompiledFunction] = field(init=False, default=None)
    """Wrench rotation and one integration step, compiled together in
    :meth:`build_artifacts`."""

    _force_torque_sensor: Optional[ForceTorqueSensor] = field(init=False, default=None)
    """Live sensor annotation, resolved from :attr:`tip_link`."""

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
        Compile the live wrench, rotated into the goal frame and integrated by one step of
        the virtual dynamics, into :attr:`_compiled_step`.

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
        measured_force = (
            goal_T_sensor.to_rotation_matrix() @ self._force_torque_sensor.force
        )
        control_dt = context.qp_controller_config.control_dt

        next_positions, next_velocities = [], []
        for axis in range(len(AXES)):
            mass, damping = self.mass[axis], self.damping[axis]
            stiffness = self.stiffness[axis]
            position, velocity = position_variables[axis], velocity_variables[axis]
            force_error = measured_force[axis] - self.desired_force[axis]
            next_velocity = (
                mass * velocity + (force_error - stiffness * position) * control_dt
            ) / (mass + damping * control_dt + stiffness * control_dt**2)
            next_positions.append(position + next_velocity * control_dt)
            next_velocities.append(next_velocity)

        self._compiled_step = vstack(next_positions + next_velocities).compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                self._force_torque_sensor.wrench_data.variables,
                position_variables + velocity_variables,
            ),
            sparse=False,
        )
