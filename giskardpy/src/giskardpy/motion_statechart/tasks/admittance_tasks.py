from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import NonPositiveVirtualMassError
from giskardpy.motion_statechart.graph_node import NodeArtifacts
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

# %% yielding to contact force


@dataclass(eq=False, repr=False)
class AdmittanceCartesianTrajectory(CartesianPositionTrajectory):
    """
    Follow a trajectory while yielding to contact force.

    The tip is pulled towards the trajectory point offset by a virtual
    mass-damper-spring, driven by the difference between the measured and the desired
    contact force. The offset carries across the whole trajectory, so the press stays
    settled while the tool travels from waypoint to waypoint.

    .. note:: Without stiffness the offset also seeks contact on its own, so the
        commanded surface height need not be exact: out of contact the whole desired
        force is an error and the tool drifts towards the surface until it touches.
    """

    desired_force: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Contact force the admittance balances, in the goal frame, in N.
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
    """

    stiffness: Vector3 = field(default_factory=Vector3, kw_only=True)
    """
    Virtual stiffness per axis, in N/m.

    Zero lets the offset persist once the force is balanced, instead of pulling it back
    to the nominal trajectory.
    """

    maximum_offset: Vector3 = field(
        default_factory=lambda: Vector3(x=0.02, y=0.02, z=0.02), kw_only=True
    )
    """
    How far the offset may take the tool from the trajectory, per axis, in m.

    .. warning:: Without a bound the offset is a pure integrator: an unbalanced desired
        force walks the goal metres away from the trajectory while the arm catches up.
    """

    lead_time: float = field(default=0.0, kw_only=True)
    """
    How far ahead of the offset the tracked point is placed, in s.

    Compensates the delay between commanding a pose and the tool reaching it; zero hands
    the offset on as it stands.
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
    Wrench rotation and one integration step, compiled together.
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

    @property
    def offset_from_trajectory(self) -> np.ndarray:
        return self._state[:3] + self.lead_time * self._state[3:]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Compile the admittance before the trajectory constrains the tip to the offset
        point.
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
                self._admittance_position,
                self._state[:3] + self.lead_time * self._state[3:],
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

        .. note:: Every parameter group the step compiles against is of fixed size, so
            it stays independent of how many nodes the chart holds.

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
        self._compiled_step = self._yielding_step(
            measured_force=measured_force,
            positions=position_variables,
            velocities=velocity_variables,
            control_dt=context.qp_controller_config.control_time_step.total_seconds(),
        ).compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                self._force_torque_sensor.wrench_data.variables,
                position_variables + velocity_variables,
            ),
            sparse=False,
        )

    def _yielding_step(
        self,
        measured_force: Vector3,
        positions: List[FloatVariable],
        velocities: List[FloatVariable],
        control_dt: float,
    ) -> sm.Vector:
        """
        One step of the virtual dynamics, per axis: ``mass * acceleration + damping *
        velocity + stiffness * position = measured_force - desired_force``.

        .. warning:: The damping and stiffness terms are taken implicitly, which is
            stable for any positive parameters; taken explicitly the integration
            diverges once ``control_dt * damping / mass`` exceeds 2.

        :return: The next offsets followed by the next rates, stacked.
        """
        next_positions, next_velocities = [], []
        for axis in range(len(AXES)):
            position, velocity = positions[axis], velocities[axis]
            error = measured_force[axis] - self.desired_force[axis]
            next_velocity = (
                self.mass[axis] * velocity
                + (error - self.stiffness[axis] * position) * control_dt
            ) / (
                self.mass[axis]
                + self.damping[axis] * control_dt
                + self.stiffness[axis] * control_dt**2
            )
            next_positions.append(
                limit(
                    position + next_velocity * control_dt,
                    -self.maximum_offset[axis],
                    self.maximum_offset[axis],
                )
            )
            next_velocities.append(next_velocity)
        return vstack(next_positions + next_velocities)
