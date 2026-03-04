from __future__ import annotations

import inspect
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Callable, Type, Optional

from krrood.adapters.json_serializer import list_like_classes, leaf_types
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.factories import variable, and_
from krrood.entity_query_language.query.match import UnderspecifiedVariable

from krrood.probabilistic_knowledge.probable_variable import (
    QueryToRandomEventTranslator,
)
from random_events.set import Set
from random_events.variable import Symbolic

from typing_extensions import Any

from random_events.variable import variable_from_name_and_type


@dataclass
class CallableAndKwargs:
    """
    A hierarchy of callables and their keyword arguments.
    This behaves like a factory for underspecified method calls where the missing parameters are indicated by Ellipsis.
    """

    callable: Callable
    """
    The callable to call when all parameters are specified.
    """

    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to pass to the callable.
    """

    def __deepcopy__(self, memo):
        return self.__class__(
            self.callable,
            {name: deepcopy(value) for name, value in self.kwargs.items()},
        )

    @cached_property
    def flat_assignments(
        self,
    ) -> Dict[VariableThatAppearsKwargsAndWhereConditions, Any]:
        """
        :return: A dictionary mapping all variables mentioned in the CallableAndKwargs to their values.
        """
        result = {}

        symbolic_access_for_kwargs = variable(self.callable, [])
        symbolic_access_for_where = variable(self.callable, [])
        symbolic_access = VariableThatAppearsKwargsAndWhereConditions(
            symbolic_access_for_kwargs, symbolic_access_for_where
        )

        self._flat_assignments(symbolic_access, result)
        return result

    def get_type_hint_of_keyword_argument(self, name: str):
        hints = typing.get_type_hints(
            self.callable,
            globalns=getattr(self.callable, "__globals__", None),
            localns=None,
            include_extras=True,  # keeps Annotated[...] / other extras if you use them
        )
        return hints.get(name)

    def _flat_assignments(
        self,
        symbolic_access: VariableThatAppearsKwargsAndWhereConditions,
        result: Dict,
    ):
        """
        Recursively extract all variables and their values from the CallableAndKwargs.

        :param symbolic_access: The current symbolic access to the variables.
        :param result: The dictionary to store the extracted variables and their values.
        """
        symbolic_access.variable_in_kwargs = symbolic_access.variable_in_kwargs.kwargs
        for key, value in self.kwargs.items():

            # update access patterns
            current_symbolic_access = symbolic_access.apply_attribute_access(key)
            current_symbolic_access.type_hint = self.get_type_hint_of_keyword_argument(
                key
            )

            if isinstance(value, list_like_classes):
                # handle list like classes by wrapping the index access
                for index, element in enumerate(value):

                    # update access pattern for the current index
                    current_symbolic_access_index = (
                        current_symbolic_access.apply_index_access(index)
                    )
                    current_symbolic_access.type_hint = (
                        self.get_type_hint_of_keyword_argument(key)
                    )  # get the type hint from the signature of the function
                    if isinstance(element, CallableAndKwargs):
                        element._flat_assignments(
                            current_symbolic_access_index,
                            result,
                        )
                    else:
                        result[current_symbolic_access_index] = element

            elif isinstance(value, CallableAndKwargs):
                value._flat_assignments(
                    current_symbolic_access,
                    result,
                )
            else:
                result[current_symbolic_access] = value

    def apply_assignments(
        self, bindings: Dict[VariableThatAppearsKwargsAndWhereConditions, Any]
    ):
        """
        Apply the given bindings to the CallableAndKwargs instance.
        This updates the `kwarg` attribute in-place.

        :param bindings: A dictionary mapping symbolic accesses from this instance to their values.
        """
        for variable, value in bindings.items():
            variable.variable_in_kwargs._set_external_instance_value_(self, value)

    def construct_instance(self):
        """
        Construct a python object from the CallableAndKwargs instance.

        ..note:: This method may work with ellipsis, but it's not guaranteed to work with all types.

        :return: The constructed object.
        """
        constructed_kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list_like_classes):
                constructed_kwargs[key] = type(value)(
                    (
                        element.construct_instance()
                        if isinstance(element, CallableAndKwargs)
                        else element
                    )
                    for element in value
                )

            elif isinstance(value, CallableAndKwargs):
                constructed_kwargs[key] = value.construct_instance()
            else:
                constructed_kwargs[key] = value
        return self.callable(**constructed_kwargs)


@dataclass
class UnderspecifiedToCallableAndKwargsTranslator:
    """
    Extract CallableAndKwargs object from an UnderspecifiedVariable.
    """

    statement: UnderspecifiedVariable
    """
    The UnderspecifiedVariable to translate.
    """

    def translate(self) -> CallableAndKwargs:
        """
        :return: A CallableAndKwargs object extracted from the UnderspecifiedVariable.
        """
        return self._translate(self.statement)

    def _translate(
        self,
        statement: UnderspecifiedVariable,
    ) -> CallableAndKwargs:
        """
        Extract CallableAndKwargs object from an UnderspecifiedVariable recoursively.
        :param statement: The statement to translate.
        :return: A CallableAndKwargs object extracted from the statement.
        """

        current_callable = statement.factory

        kwargs = dict()

        for key, argument in statement.kwargs.items():

            if isinstance(argument, list_like_classes):

                kwargs[key] = type(argument)(
                    (
                        self._translate(element)
                        if isinstance(element, UnderspecifiedVariable)
                        else element
                    )
                    for element in argument
                )

            elif isinstance(argument, UnderspecifiedVariable):
                kwargs[key] = self._translate(argument)

            else:
                kwargs[key] = argument
        return CallableAndKwargs(current_callable, kwargs)


@dataclass
class VariableThatAppearsKwargsAndWhereConditions:
    variable_in_kwargs: MappedVariable
    variable_in_where_conditions: MappedVariable
    type_hint: Optional[Type] = None

    def __hash__(self):
        return hash(self.variable_in_kwargs)

    def apply_attribute_access(self, attribute: str):
        return self.__class__(
            self.variable_in_kwargs[attribute],
            getattr(self.variable_in_where_conditions, attribute),
        )

    def apply_index_access(self, index: Any):
        return self.__class__(
            self.variable_in_kwargs[index],
            self.variable_in_where_conditions[index],
        )


@dataclass
class Parameters:
    statement: UnderspecifiedVariable

    _factory_compiler: UnderspecifiedToCallableAndKwargsTranslator = field(init=False)
    _random_event_compiler: QueryToRandomEventTranslator = field(init=False)

    def __post_init__(self):
        self._factory_compiler = UnderspecifiedToCallableAndKwargsTranslator(
            self.statement
        )
        self._random_event_compiler = QueryToRandomEventTranslator(
            and_(*self.statement._where_expression)
        )

    @property
    def random_event_variables(self):
        for (
            variable,
            value,
        ) in self._factory_compiler.translate().flat_assignments.items():
            if isinstance(value, type(Ellipsis)):
                random_events_variable = variable_from_name_and_type(
                    variable.variable_in_where_conditions.name, variable.type_hint
                )
            elif isinstance(value, leaf_types):
                random_events_variable = variable_from_name_and_type(
                    variable.variable_in_where_conditions.name, type(value)
                )
            elif isinstance(value, SymbolicExpression):
                random_events_variable = self.random_event_variable_from_expression(
                    variable.variable_in_where_conditions.name, value
                )
            return random_events_variable

    def random_event_variable_from_expression(
        self, name: str, expression: SymbolicExpression
    ):
        variable = Symbolic(name, Set.from_iterable(expression.tolist()))
        return variable
