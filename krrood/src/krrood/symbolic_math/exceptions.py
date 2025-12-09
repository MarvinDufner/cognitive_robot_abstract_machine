from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    List,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from krrood.symbolic_math.symbolic_math import FloatVariable


class SymbolicMathError(Exception):
    pass


@dataclass
class WrongDimensionsError(SymbolicMathError):
    expected_dimensions: Tuple[int, int]
    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected {self.expected_dimensions} dimensions, but got {self.actual_dimensions}."
        super().__init__(msg)


@dataclass
class NotScalerError(WrongDimensionsError):
    expected_dimensions: Tuple[int, int] = field(default=(1, 1), init=False)


@dataclass
class NotSquareMatrixError(SymbolicMathError):
    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected a square matrix, but got {self.actual_dimensions} dimensions."
        super().__init__(msg)


@dataclass
class HasFreeVariablesError(SymbolicMathError):
    """
    Raised when an operation can't be performed on an expression with free variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation can't be performed on expression with free variables: {self.variables}."
        super().__init__(msg)


class ExpressionEvaluationError(SymbolicMathError): ...


@dataclass
class WrongNumberOfArgsError(ExpressionEvaluationError):
    expected_number_of_args: int
    actual_number_of_args: int

    def __post_init__(self):
        msg = f"Expected {self.expected_number_of_args} arguments, but got {self.actual_number_of_args}."
        super().__init__(msg)


@dataclass
class DuplicateVariablesError(SymbolicMathError):
    """
    Raised when duplicate variables are found in an operation that requires unique variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation failed due to duplicate variables: {self.variables}. All variables must be unique."
        super().__init__(msg)
