import operator
from collections import deque
from dataclasses import dataclass
from typing import assert_never, List, Dict

import numpy as np

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import OR, AND
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.query import Entity
from krrood.probabilistic_knowledge.exceptions import (
    WhereExpressionNotInDisjunctiveNormalForm,
)
from random_events.interval import closed_open, closed, open
from random_events.product_algebra import Event, SimpleEvent


@dataclass
class QueryToRandomEventTranslator:
    """
    Class that translates a query into a random event.
    Requires that the query is in disjunctive normal form.

    Check the documentation of `is_disjunctive_normal_form` for more information.

    """

    query: Entity
    """
    The query in disjunctive normal form to translate.
    """

    def __post_init__(self):
        if not is_disjunctive_normal_form(self.query):
            raise WhereExpressionNotInDisjunctiveNormalForm(
                self.query._where_expression_
            )

    def translate(self) -> Event:
        """
        :return: The random event that corresponds to the query.
        """

        simple_events = []

        # Traverse the logical tree starting from the conditions root
        root = self.query._conditions_root_

        queue = deque([root])
        while queue:
            expression = queue.popleft()

            if isinstance(expression, OR):
                queue.extend(expression._children_)
                continue

            elif isinstance(expression, AND):
                simple_event = self._translate_conjunction(expression)
            elif isinstance(expression, Comparator):
                simple_event = SimpleEvent(
                    {v.variable: v.variable.domain for v in self.all_variables}
                )
                self._translate_comparators(
                    self._object_access_variable_from_comparator(expression),
                    [expression],
                    simple_event,
                )
            else:
                assert_never(expression)
            simple_events.append(simple_event)
        return Event(*simple_events)

    def _translate_conjunction(self, expression: AND) -> SimpleEvent:
        """
        Translate a conjunction expression into a random event.
        The conjunction must not contain any disjunctions anymore.

        :param expression: The conjunction expression to translate.
        :return: The random event corresponding to the conjunction.
        """
        result = SimpleEvent()

        # check that it is always a comparison between a variable and a literal
        for variable, comparators in self.comparators_grouped_by_variable(
            expression
        ).items():
            self._translate_comparators(variable, comparators, result)

        return result

    @symbolic_function
    def _object_access_variable_from_comparator(
        self, comparator: Comparator
    ) -> ObjectAccessVariable:
        """
        Create an ObjectAccessVariable from a comparator.
        Requires that the comparator's left operand is an attribute access like operation.

        :param comparator: The comparator to extract the variable from.
        :return: The ObjectAccessVariable corresponding to the comparator's left operand.
        """
        assert isinstance(comparator.left, AttributeAccessLike)
        return ObjectAccessVariable.from_attribute_access_and_type(
            comparator.left, comparator.left._type_
        )

    @property
    def all_variables(self) -> List[ObjectAccessVariable]:
        root = self.query._conditions_root_
        return list(self.comparators_grouped_by_variable(root).keys())

    def comparators_grouped_by_variable(
        self, expression: SymbolicExpression
    ) -> Dict[ObjectAccessVariable, List[Comparator]]:
        """
        Group comparators by their variable given an expression.

        :param expression: The expression where all comparators in the descendents should be grouped by variables.
        :return: A dictionary mapping ObjectAccessVariables to lists of their corresponding comparators.
        """

        # Collect all Comparator descendants and group them by their accessed variable
        grouped: Dict[ObjectAccessVariable, List[Comparator]] = {}
        for expr in expression._descendants_:
            if not isinstance(expr, Comparator):
                continue
            key = self._object_access_variable_from_comparator(expr)
            grouped.setdefault(key, []).append(expr)
        return grouped

    def _translate_comparators(
        self,
        variable: ObjectAccessVariable,
        comparators: List[Comparator],
        result: SimpleEvent,
    ) -> None:
        """
        Translate comparators for a given variable into a random event in-place.

        :param variable: The variable for which to translate comparators.
        :param comparators: The comparators to translate.
        :param result: The random event to update in-place.
        :return: None
        """

        result[variable.variable] = variable.variable.domain
        for comparator in comparators:

            if isinstance(comparator.right._value_, type(Ellipsis)):
                continue

            match comparator.operation:
                case operator.eq:
                    self._translate_eq(comparator, variable, result)
                case operator.ne:
                    self._translate_ne(comparator, variable, result)
                case operator.gt:
                    self._translate_gt(comparator, variable, result)
                case operator.lt:
                    self._translate_lt(comparator, variable, result)
                case operator.ge:
                    self._translate_ge(comparator, variable, result)
                case operator.le:
                    self._translate_le(comparator, variable, result)
                case _:
                    assert_never(comparator.operation)

    def _translate_eq(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[
            object_access_variable.variable
        ] &= object_access_variable.variable.make_value(comparator.right._domain_[0])

    def _translate_ne(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[
            object_access_variable.variable
        ] &= object_access_variable.variable.make_value(
            comparator.right._domain_[0]
        ).complement()

    def _translate_gt(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[object_access_variable.variable] &= open(
            comparator.right._domain_[0], np.inf
        )

    def _translate_lt(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[object_access_variable.variable] &= closed_open(
            -np.inf,
            comparator.right._domain_[0],
        )

    def _translate_le(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[object_access_variable.variable] &= closed(
            -np.inf,
            comparator.right._domain_[0],
        )

    def _translate_ge(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[object_access_variable.variable] &= closed(
            comparator.right._domain_[0],
            np.inf,
        )


def is_disjunctive_normal_form(query: Entity) -> bool:
    """
    Checks if the given query is disjunctive normal form (DNF).

    A query is in DNF if the following 3 statements are true:
    1. All its comparators are literal comparators, i.e. comparators between one variable and one literal
    2. All of its conjunctions (AND statements) only have literal comparators as children
    3. There is at most one disjunction (OR statement) which has to be at the root.

    Example:
        (x > 3) is DNF

        (x > 3) & (y < 5) is DNF

        (x > 3) | (y < 5) is DNF

        (x > 3) | ((y > 5) & (z < 2)) is DNF

        (x > 3) & ((y > 5) | (z < 2)) is not DNF

    :param query: The query to check
    :return: True if the query is disjunctive normal form, False otherwise
    """

    condition_root = query._conditions_root_

    return (
        is_disjunction_of_conjunction_of_literal_comparators(condition_root)
        or is_conjunction_of_literal_comparators(condition_root)
        or is_literal_comparator(condition_root)
    )


def is_disjunction_of_conjunction_of_literal_comparators(expression: OR) -> bool:
    """
    Checks if the given expression is a disjunction of conjunctions of literal comparators.

    :param expression: The expression to check.
    :return: True if the expression is a disjunction of conjunctions of literal comparators, False otherwise.
    """
    if not isinstance(expression, OR):
        return False
    for child in expression._children_:
        if not (
            is_disjunction_of_conjunction_of_literal_comparators(child)
            or is_conjunction_of_literal_comparators(child)
            or is_literal_comparator(child)
        ):
            return False
    return True


def is_conjunction_of_literal_comparators(expression: AND) -> bool:
    """
    Checks if the given expression is a conjunction of literal comparators.

    :param expression: The expression to check.
    :return: True if the expression is a conjunction of literal comparators, False otherwise.
    """
    if not isinstance(expression, AND):
        return False
    for child in expression._children_:
        if not (
            is_conjunction_of_literal_comparators(child) or is_literal_comparator(child)
        ):
            return False

    return True


def is_literal_comparator(expression: Comparator) -> bool:
    """
    Checks if the given expression is a literal comparator.

    :param expression: The expression to check.
    :return: True if the expression is a literal comparator, False otherwise.
    """
    if not isinstance(expression, Comparator):
        return False
    if not isinstance(expression.left, MappedVariable):
        return False
    if not isinstance(expression.right, Literal):
        return False
    return True
