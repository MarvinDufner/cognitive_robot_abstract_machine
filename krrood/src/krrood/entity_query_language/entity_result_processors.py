from __future__ import annotations

from typing_extensions import Optional, Union, Callable, Any

from .result_quantification_constraint import (
    ResultQuantificationConstraint,
)
from .symbolic import (
    An,
    The,
    Selectable,
    Max,
    Min,
    Sum,
    Average,
    Count,
    QueryObjectDescriptor,
    QuantifierBuilder,
)
from .utils import T


def an(
    entity: Union[T, QueryObjectDescriptor],
    quantification: Optional[ResultQuantificationConstraint] = None,
) -> Union[T, QueryObjectDescriptor]:
    """
    Select all values satisfying the given entity description.

    :param entity: An entity or a set expression to quantify over.
    :param quantification: Optional quantification constraint.
    :return: The entity with the quantifier applied.
    """
    return entity._quantify_(An, quantification_constraint=quantification)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(
    entity: Union[T, QueryObjectDescriptor],
) -> Union[T, QueryObjectDescriptor]:
    """
    Select the unique value satisfying the given entity description.

    :param entity: An entity or a set expression to quantify over.
    :return: The entity with the quantifier applied.
    """
    return entity._quantify_(The)


def max(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Max[T]]:
    """
    Maps the variable values to their maximum value.

    :param variable: The variable for which the maximum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Max object that can be evaluated to find the maximum value.
    """
    return Max(variable, _key_function_=key, _default_value_=default, _distinct_=distinct)


def min(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Min[T]]:
    """
    Maps the variable values to their minimum value.

    :param variable: The variable for which the minimum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Min object that can be evaluated to find the minimum value.
    """
    return Min(variable, _key_function_=key, _default_value_=default, _distinct_=distinct)


def sum(
    variable: Union[T, Selectable[T]],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Sum]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return Sum(variable, _key_function_=key, _default_value_=default, _distinct_=distinct)


def average(
    variable: Union[Selectable[T], Any],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Average]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return Average(
        variable, _key_function_=key, _default_value_=default, _distinct_=distinct
    )


def count(
    variable: Optional[Selectable[T]] = None, distinct: bool = False
) -> Union[T, Count[T]]:
    """
    Count the number of values produced by the given variable.

    :param variable: The variable for which the count is calculated, if not given, the count of all results (by group)
     is returned.
    :param distinct: Whether to only consider distinct values.
    :return: A Count object that can be evaluated to count the number of values.
    """
    return Count(variable, _distinct_=distinct)
