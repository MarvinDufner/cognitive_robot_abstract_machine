from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.utils.decorators import dataclass


@dataclass
class ThreadedPayloadMonitor(MotionStatechartNode, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.
    Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
    it's done.
    Calls __call__ in a separate thread. Use for expensive operations
    """

    state: ObservationStateValues = field(
        init=False, default=ObservationStateValues.UNKNOWN
    )

    @abc.abstractmethod
    def __call__(self):
        pass


@dataclass
class LocalMinimumReached(MotionStatechartNode):
    min_cut_off: float = 0.01
    max_cut_off: float = 0.06
    joint_convergence_threshold: float = 0.01
    windows_size: int = 1

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        ref = []
        symbols = []
        for dof in context.world.active_degrees_of_freedom:
            velocity_limit = dof.upper_limits.velocity
            velocity_limit *= self.joint_convergence_threshold
            velocity_limit = min(
                max(self.min_cut_off, velocity_limit), self.max_cut_off
            )
            ref.append(velocity_limit)
            symbols.append(dof.variables.velocity)
        ref = cas.Expression(ref)
        vel_symbols = cas.Expression(symbols)

        dt = (
            context.qp_controller_config.control_dt
            or context.qp_controller_config.mpc_dt
        )
        traj_longer_than_1_sec = context.control_cycle_variable * dt > 1
        artifacts.observation = cas.trinary_logic_and(
            traj_longer_than_1_sec, cas.logic_all(cas.abs(vel_symbols) < ref)
        )
        return artifacts


@dataclass
class TimeAbove(MotionStatechartNode):
    threshold: float = field(kw_only=True)

    def __post_init__(self):
        traj_length_in_sec = context.time_symbol
        condition = traj_length_in_sec > self.threshold
        self.observation_expression = condition


@dataclass
class Alternator(MotionStatechartNode):
    mod: int = 2

    def __post_init__(self):
        time = context.time_symbol
        expr = cas.fmod(cas.floor(time), self.mod) == 0
        self.observation_expression = expr
