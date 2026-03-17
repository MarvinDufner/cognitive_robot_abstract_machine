from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, Field
from functools import cached_property

from typing_extensions import (
    Any,
    Callable,
    TypeVar,
    Dict,
    List,
    Union,
    Iterable,
    Generator,
)

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    variable,
    a,
    set_of,
    evaluate_condition,
)
from ...datastructures.dataclasses import Context
from pycram.designator import DesignatorDescription
from pycram.failures import PlanFailure, ConditionNotSatisfied

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(DesignatorDescription, ABC):
    _pre_perform_callbacks = []
    _post_perform_callbacks = []

    def perform(self) -> Any:
        """
        Full execution: pre-check, plan, post-check
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        for pre_cb in self._pre_perform_callbacks:
            pre_cb(self)

        if self.plan.context.evaluate_conditions:
            self.evaluate_pre_condition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            pass

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Symbolic plan. Should only call motions or sub-actions.
        """
        pass

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @classmethod
    def pre_perform(cls, func) -> Callable:
        cls._pre_perform_callbacks.append(func)
        return func

    @classmethod
    def post_perform(cls, func) -> Callable:
        cls._post_perform_callbacks.append(func)
        return func

    @cached_property
    def bound_variables(self) -> Dict[T, Variable[T] | T]:
        return self._create_variables()

    def _create_variables(self) -> Dict[str, Variable[T] | T]:
        """
        Creates krrood variables for all parameter of this action

        :return: A dict with action parameters as keys and variables as values.
        """
        return {
            f.name: variable(
                type(getattr(self, f.name)),
                ([getattr(self, f.name)]),
            )
            for f in self.fields
        }

    def evaluate_pre_condition(self) -> bool:
        condition = self.pre_condition(
            self.bound_variables,
            self.context,
            self.action_parameter,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(True, self.__class__, condition)

    def evaluate_post_condition(self) -> bool:
        condition = self.post_condition(
            self.bound_variables,
            self.context,
            self.action_parameter,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(False, self.__class__, condition)


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T, ...]
