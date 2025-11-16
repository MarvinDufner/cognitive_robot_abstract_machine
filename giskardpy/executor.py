from dataclasses import dataclass, field, InitVar

import numpy as np
from typing_extensions import Optional

from giskardpy.data_types.exceptions import (
    EmptyProblemException,
    NoQPControllerConfigException,
)
from giskardpy.model.better_pybullet_syncer import BulletCollisionDetector
from giskardpy.model.collision_world_syncer import (
    CollisionWorldSynchronizer,
    CollisionCheckerLib,
)
from giskardpy.model.collisions import NullCollisionDetector
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
)
from semantic_digital_twin.world import World


@dataclass
class Executor:
    world: World
    controller_config: Optional[QPControllerConfig] = None
    collision_checker: InitVar[CollisionCheckerLib] = field(
        default=CollisionCheckerLib.none
    )

    # %% init False
    motion_statechart: MotionStatechart = field(init=False)
    collision_scene: Optional[CollisionWorldSynchronizer] = field(
        default=None, init=False
    )
    auxiliary_variable_manager: AuxiliaryVariableManager = field(
        default_factory=AuxiliaryVariableManager, init=False
    )
    qp_controller: Optional[QPController] = field(default=None, init=False)

    _control_cycles: int = field(init=False)
    _control_cycles_variable: AuxiliaryVariable = field(init=False)

    _time: float = field(init=False)
    _time_variable: AuxiliaryVariable = field(init=False)

    def __post_init__(self, collision_checker: CollisionCheckerLib):
        if collision_checker == CollisionCheckerLib.bpb:
            collision_detector = BulletCollisionDetector(_world=self.world)
        else:
            collision_detector = NullCollisionDetector(_world=self.world)

        self.collision_scene = CollisionWorldSynchronizer(
            world=self.world,
            robots=self.world.get_semantic_annotations_by_type(AbstractRobot),
            collision_detector=collision_detector,
        )

    def _create_control_cycles_variable(self):
        self._control_cycles_variable = (
            self.auxiliary_variable_manager.create_float_variable(
                PrefixedName("control_cycles"), lambda: self._control_cycles
            )
        )

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self._control_cycles = -1
        self._create_control_cycles_variable()
        self.motion_statechart.compile(self.build_context)
        self._compile_qp_controller(self.controller_config)

    @property
    def build_context(self) -> BuildContext:
        return BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
            qp_controller_config=self.controller_config,
            control_cycle_variable=self._control_cycles_variable,
        )

    def tick(self):
        self._control_cycles += 1
        self.collision_scene.sync()
        self.collision_scene.check_collisions()
        self.motion_statechart.tick(self.build_context)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.get_cmd(
            world_state=self.world.state.data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            external_collisions=self.collision_scene.get_external_collision_data(),
            self_collisions=self.collision_scene.get_self_collision_data(),
            auxiliary_variables=self.auxiliary_variable_manager.resolve_auxiliary_variables(),
        )
        self.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt or self.qp_controller.config.mpc_dt,
            self.qp_controller.config.max_derivative,
        )

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        """
        for i in range(timeout):
            self.tick()
            if self.motion_statechart.is_end_motion():
                return
        raise TimeoutError("Timeout reached while waiting for end of motion.")

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.world.active_degrees_of_freedom,
            key=lambda dof: self.world.state._index[dof.name],
        )
        constraint_collection = (
            self.motion_statechart.combine_constraint_collections_of_nodes()
        )
        if len(constraint_collection.constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        elif controller_config is None:
            raise NoQPControllerConfigException(
                "constraints but no controller config given."
            )
        self.qp_controller = QPController(
            config=controller_config,
            degrees_of_freedom=ordered_dofs,
            constraint_collection=constraint_collection,
            world_state_symbols=self.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            external_collision_avoidance_variables=self.collision_scene.get_external_collision_symbol(),
            self_collision_avoidance_variables=self.collision_scene.get_self_collision_symbol(),
            auxiliary_variables=self.auxiliary_variable_manager.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException(
                "Tried to compile a QPController without free variables."
            )
