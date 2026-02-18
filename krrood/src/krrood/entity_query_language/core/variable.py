from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING, is_dataclass, fields
from functools import cached_property

from typing_extensions import Generic, Type, Tuple, Any, Dict, Optional, Iterable, List, Union as TypingUnion, Union, \
    Callable

from .base_expressions import Bindings, OperationResult, SymbolicExpression, TruthValueOperator, UnaryExpression
from ..cache_data import ReEnterableLazyIterable
from ..failures import VariableCannotBeEvaluated
from ..operators.comparator import Comparator
from ..operators.set_operations import MultiArityExpressionThatPerformsACartesianProduct
from ..utils import T, merge_args_and_kwargs, convert_args_and_kwargs_into_a_hashable_key, \
    is_iterable, make_list
from ...class_diagrams.class_diagram import WrappedClass
from ...class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ...class_diagrams.wrapped_field import WrappedField
from ...symbol_graph.symbol_graph import SymbolGraph


@dataclass(eq=False, repr=False)
class Selectable(SymbolicExpression, Generic[T], ABC):
    _var_: Selectable[T] = field(init=False, default=None)
    """
    A variable that is used if the child class to this class want to provide a variable to be tracked other than 
    itself, this is specially useful for child classes that holds a variable instead of being a variable and want
     to delegate the variable behaviour to the variable it has instead.
    For example, this is the case for the ResultQuantifiers & QueryDescriptors that operate on a single selected
    variable.
    """

    _type_: Type[T] = field(init=False, default=None)
    """
    The type of the variable.
    """

    @cached_property
    def _binding_id_(self) -> int:
        return (
            self._var_._binding_id_
            if self._var_ is not None and self._var_ is not self
            else self._id_
        )

    @cached_property
    def _type__(self):
        return (
            self._var_._type_
            if self._var_ is not None and self._var_ is not self
            else None
        )

    def _process_result_(self, result: OperationResult) -> T:
        """
        Map the result to the correct output data structure for user usage.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return result.value

    @property
    def _is_iterable_(self):
        """
        Whether the selectable is iterable.

        :return: True if the selectable is iterable, False otherwise.
        """
        if self._var_ and self._var_ is not self:
            return self._var_._is_iterable_
        return False


@dataclass
class DomainMappingCacheItem:
    """
    A cache item for domain mapping creation. To prevent recreating same mapping multiple times, mapping instances are
    stored in a dictionary with a hashable key. This class is used to generate the key for the dictionary that stores
    the mapping instances.
    """

    type: Type[DomainMapping]
    """
    The type of the domain mapping.
    """
    child: CanBehaveLikeAVariable
    """
    The child of the domain mapping (i.e. the original variable on which the domain mapping is applied).
    """
    args: Tuple[Any, ...] = field(default_factory=tuple)
    """
    Positional arguments to pass to the domain mapping constructor.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Keyword arguments to pass to the domain mapping constructor.
    """

    def __post_init__(self):
        self.args = (self.child,) + self.args

    @cached_property
    def all_kwargs(self):
        return merge_args_and_kwargs(
            self.type, self.args, self.kwargs, ignore_first=True
        )

    @cached_property
    def hashable_key(self):
        return (self.type,) + convert_args_and_kwargs_into_a_hashable_key(
            self.all_kwargs
        )

    def __hash__(self):
        return hash(self.hashable_key)

    def __eq__(self, other):
        return (
                isinstance(other, DomainMappingCacheItem)
                and self.hashable_key == other.hashable_key
        )


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
    and comparison operations.
    """

    _known_mappings_: Dict[DomainMappingCacheItem, DomainMapping] = field(
        init=False, default_factory=dict
    )
    """
    A storage of created domain mappings to prevent recreating same mapping multiple times.
    """

    def _update_truth_value_(self, current_value: Any) -> None:
        """
        Updates the truth value of the variable based on the current value.

        :param current_value: The current value of the variable.
        """
        # Calculating the truth value is not always done for efficiency. The truth value is updated only when this
        # operation is a child of a TruthValueOperator.
        if isinstance(self._parent_, TruthValueOperator):
            is_true = (
                len(current_value) > 0
                if is_iterable(current_value)
                else bool(current_value)
            )
            self._is_false_ = not is_true

    def _get_domain_mapping_(
            self, type_: Type[DomainMapping], *args, **kwargs
    ) -> DomainMapping:
        """
        Retrieves or creates a domain mapping instance based on the provided arguments.

        :param type_: The type of the domain mapping to retrieve or create.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The retrieved or created domain mapping instance.
        """
        cache_item = DomainMappingCacheItem(type_, self, args, kwargs)
        if cache_item in self._known_mappings_:
            return self._known_mappings_[cache_item]
        else:
            instance = type_(**cache_item.all_kwargs)
            self._known_mappings_[cache_item] = instance
            return instance

    def _get_domain_mapping_key_(self, type_: Type[DomainMapping], *args, **kwargs):
        """
        Generates a hashable key for the given type and arguments.

        :param type_: The type of the domain mapping.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The generated hashable key.
        """
        args = (self,) + args
        all_kwargs = merge_args_and_kwargs(type_, args, kwargs, ignore_first=True)
        return convert_args_and_kwargs_into_a_hashable_key(all_kwargs)

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        return self._get_domain_mapping_(Attribute, name, self._type__)

    def __getitem__(self, key) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Index, key)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Call, args, kwargs)

    def __eq__(self, other) -> Comparator:
        return Comparator(self, other, operator.eq)

    def __ne__(self, other) -> Comparator:
        return Comparator(self, other, operator.ne)

    def __lt__(self, other) -> Comparator:
        return Comparator(self, other, operator.lt)

    def __le__(self, other) -> Comparator:
        return Comparator(self, other, operator.le)

    def __gt__(self, other) -> Comparator:
        return Comparator(self, other, operator.gt)

    def __ge__(self, other) -> Comparator:
        return Comparator(self, other, operator.ge)

    def __hash__(self):
        return super().__hash__()


@dataclass(eq=False, repr=False)
class Variable(CanBehaveLikeAVariable[T]):
    """
    A Variable that queries will assign. The Variable produces results of type `T`.
    """

    _type_: Union[Type, Callable] = field(default=MISSING)
    """
    The result type of the variable. (The value of `T`)
    """
    _name__: str
    """
    The name of the variable.
    """
    _domain_source_: Optional[DomainType] = field(
        default=None, kw_only=True, repr=False
    )
    """
    An optional source for the variable domain. If not given, the global cache of the variable class type will be used
    as the domain, or if kwargs are given the type and the kwargs will be used to inference/infer new values for the
    variable.
    """
    _domain_: ReEnterableLazyIterable = field(
        default_factory=ReEnterableLazyIterable, kw_only=True, repr=False
    )
    """
    The iterable domain of values for this variable.
    """
    _is_inferred_: bool = field(default=False, repr=False)
    """
    Whether this variable domain is inferred or not.
    """

    def __post_init__(self):
        super().__post_init__()

        if self._domain_source_ is not None:
            self._update_domain_(self._domain_source_)

        self._var_ = self

    def _update_domain_(self, domain):
        """
        Set the domain and ensure it is a lazy re-enterable iterable.
        """
        if isinstance(domain, CanBehaveLikeAVariable):
            self._update_children_(domain)
        if isinstance(domain, (ReEnterableLazyIterable, CanBehaveLikeAVariable)):
            self._domain_ = domain
            return
        if not is_iterable(domain):
            domain = [domain]
        self._domain_.set_iterable(domain)

    def _evaluate__(
            self,
            sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        A variable either is already bound in sources by other constraints (Symbolic Expressions).,
        or will yield from current domain if exists,
        or has no domain and will instantiate new values by constructing the type if the type is given.
        """

        if self._domain_source_ is not None:
            yield from self._iterator_over_domain_values_(sources)
        elif self._is_inferred_:
            # Means that the variable gets its values from conclusions only.
            return
        else:
            raise VariableCannotBeEvaluated(self)

    def _iterator_over_domain_values_(
            self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Iterate over the values in the variable's domain, yielding OperationResult instances.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        if isinstance(self._domain_, CanBehaveLikeAVariable):
            yield from self._iterator_over_variable_domain_values_(sources)
        else:
            yield from self._iterator_over_iterable_domain_values_(sources)

    def _iterator_over_variable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is another variable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for domain in self._domain_._evaluate_(sources, parent=self):
            for v in domain.value:
                bindings = {**sources, **domain.bindings, self._binding_id_: v}
                yield self._build_operation_result_and_update_truth_value_(bindings)

    def _iterator_over_iterable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is an iterable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for v in self._domain_:
            bindings = {**sources, self._binding_id_: v}
            yield self._build_operation_result_and_update_truth_value_(bindings)

    def _build_operation_result_and_update_truth_value_(
            self, bindings: Bindings
    ) -> OperationResult:
        """
        Build an OperationResult instance and update the truth value based on the bindings.

        :param bindings: The bindings of the result.
        :return: The OperationResult instance with updated truth value.
        """
        self._update_truth_value_(bindings[self._binding_id_])
        return OperationResult(bindings, self._is_false_, self)

    def _replace_child_field_(
            self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    @property
    def _name_(self):
        return self._name__

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        return [self]

    @property
    def _is_iterable_(self):
        return is_iterable(next(iter(self._domain_), None))


@dataclass(eq=False, init=False, repr=False)
class Literal(Variable[T]):
    """
    Literals are variables that are not constructed by their type but by their given data.
    """

    def __init__(
            self,
            data: Any,
            name: Optional[str] = None,
            type_: Optional[Type] = None,
            wrap_in_iterator: bool = True,
    ):
        original_data = data
        if wrap_in_iterator:
            data = [data]
        if not type_:
            original_data_lst = make_list(original_data)
            first_value = original_data_lst[0] if len(original_data_lst) > 0 else None
            type_ = type(first_value) if first_value else None
        if name is None:
            if type_:
                name = type_.__name__
            else:
                if isinstance(data, Selectable):
                    name = data._name_
                else:
                    name = type(original_data).__name__
        super().__init__(_name__=name, _type_=type_, _domain_source_=data)


@dataclass(eq=False, repr=False)
class InstantiatedVariable(MultiArityExpressionThatPerformsACartesianProduct, Variable[T]):
    """
    A variable which does not have an explicit domain, but creates new instances using the `_type_` and `_kwargs_`
    that are provided. The `_kwargs_` are variables that can be used to generate combinations of bindings to create
    instances for each combination. By definition this variable is inferred. It also represents Predicates and symbolic
    functions.
    """
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The properties of the variable as keyword arguments.
    """
    _child_vars_: Dict[str, SymbolicExpression] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_ dictionary. 
    """
    _child_var_id_name_map_: Dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable ids to their names. 
    """

    def __post_init__(self):
        self._is_inferred_ = True
        Variable.__post_init__(self)
        self._update_child_vars_from_kwargs_()
        self._operation_children_ = tuple(self._child_vars_.values())
        # This is done here as it uses `_operation_children_`
        MultiArityExpressionThatPerformsACartesianProduct.__post_init__(self)

    def _update_child_vars_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for k, v in self._kwargs_.items():
            self._child_vars_[k] = v if isinstance(v, SymbolicExpression) else Literal(v, name=k)
            self._child_var_id_name_map_[self._child_vars_[k]._binding_id_] = k

    def _evaluate__(
            self,
            sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from self._instantiate_using_child_vars_and_yield_results_(sources)

    def _instantiate_using_child_vars_and_yield_results_(
            self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the child variables values.
        """
        for result in self._evaluate_product_(sources):
            # Build once: unwrapped hashed kwargs for already provided child vars
            kwargs = {
                self._child_var_id_name_map_[id_]: v for id_, v in result.bindings.items() if
                id_ in self._child_var_id_name_map_
            }
            instance = self._type_(**kwargs)
            yield self._process_output_and_update_values_(instance, result.bindings)

    def _process_output_and_update_values_(
            self, instance: Any, bindings: Bindings
    ) -> OperationResult:
        """
        Process the predicate/variable instance and get the results.

        :param instance: The created instance.
        :param bindings: The bindings from the child variables.
        :return: The results' dictionary.
        """
        values = {self._binding_id_: instance} | bindings
        return self._build_operation_result_and_update_truth_value_(values)

    def _replace_child_field_(
            self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        for k, v in self._child_vars_.items():
            if v is old_child:
                self._child_vars_[k] = new_child
                break

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        variables = [self]
        for v in self._child_vars_.values():
            variables.extend(v._all_variable_instances_)
        return variables


@dataclass(eq=False, repr=False)
class DomainMapping(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """

    _child_: CanBehaveLikeAVariable[T]
    """
    The child expression to apply the domain mapping to.
    """

    def __post_init__(self):
        super().__post_init__()
        self._var_ = self

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_

    @cached_property
    def _type_(self):
        return self._child_._type_

    def _evaluate__(
            self,
            sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Apply the domain mapping to the child's values.
        """

        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result, mapped_value
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
            for mapped_value in self._apply_mapping_(child_result.value)
        )

    def _build_operation_result_and_update_truth_value_(
            self, child_result: OperationResult, current_value: Any
    ) -> OperationResult:
        """
        Set the current truth value of the operation result, and build the operation result to be yielded.

        :param child_result: The current result from the child operation.
        :param current_value: The current value of this operation that is derived from the child result.
        :return: The operation result.
        """
        self._update_truth_value_(current_value)
        return OperationResult(
            {**child_result.bindings, self._binding_id_: current_value},
            self._is_false_,
            self,
        )

    @abstractmethod
    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass


@dataclass(eq=False, repr=False)
class Attribute(DomainMapping):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.

    For instance, if Body.name is called, then the attribute name is "name" and `_owner_class_` is `Body`
    """

    _attribute_name_: str
    """
    The name of the attribute.
    """

    _owner_class_: Type
    """
    The class that owns this attribute.
    """

    @property
    def _is_iterable_(self):
        if not self._wrapped_field_:
            return False
        return self._wrapped_field_.is_iterable

    @cached_property
    def _type_(self) -> Optional[Type]:
        """
        :return: The type of the accessed attribute.
        """

        if not is_dataclass(self._owner_class_):
            return None

        if self._attribute_name_ not in {f.name for f in fields(self._owner_class_)}:
            return None

        if self._wrapped_owner_class_:
            # try to get the type endpoint from a field
            try:
                return self._wrapped_field_.type_endpoint
            except (KeyError, AttributeError):
                return None
        else:
            wrapped_cls = WrappedClass(self._owner_class_)
            wrapped_cls._class_diagram = SymbolGraph().class_diagram
            wrapped_field = WrappedField(
                wrapped_cls,
                [
                    f
                    for f in fields(self._owner_class_)
                    if f.name == self._attribute_name_
                ][0],
            )
            try:
                return wrapped_field.type_endpoint
            except (AttributeError, RuntimeError):
                return None

    @cached_property
    def _wrapped_field_(self) -> Optional[WrappedField]:
        if self._wrapped_owner_class_ is None:
            return None
        return self._wrapped_owner_class_._wrapped_field_name_map_.get(
            self._attribute_name_, None
        )

    @cached_property
    def _wrapped_owner_class_(self):
        """
        :return: The owner class of the attribute from the symbol graph.
        """
        try:
            return SymbolGraph().class_diagram.get_wrapped_class(self._owner_class_)
        except ClassIsUnMappedInClassDiagram:
            return None

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield getattr(value, self._attribute_name_)

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}.{self._attribute_name_}"


@dataclass(eq=False, repr=False)
class Index(DomainMapping):
    """
    A symbolic indexing operation that can be used to access items of symbolic variables via [] operator.
    """

    _key_: Any
    """
    The key to index with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield value[self._key_]

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}[{self._key_}]"


@dataclass(eq=False, repr=False)
class Call(DomainMapping):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """

    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    """
    The arguments to call the method with.
    """
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to call the method with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        if len(self._args_) > 0 or len(self._kwargs_) > 0:
            yield value(*self._args_, **self._kwargs_)
        else:
            yield value()

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}()"


@dataclass(eq=False, repr=False)
class Flatten(DomainMapping):
    """
    Domain mapping that flattens an iterable-of-iterables into a single iterable of items.

    Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
    one solution per inner element while preserving the original bindings (e.g., the View instance),
    similar to UNNEST in SQL.
    """

    def _apply_mapping_(self, value: Iterable[Any]) -> Iterable[Any]:
        yield from value

    @cached_property
    def _name_(self):
        return f"Flatten({self._child_._name_})"

    @property
    def _is_iterable_(self):
        """
        :return: False as Flatten does not preserve the original iterable structure.
        """
        return False


def _any_of_the_kwargs_is_a_variable(bindings: Dict[str, Any]) -> bool:
    """
    :param bindings: A kwarg like dict mapping strings to objects
    :return: Rather any of the objects is a variable or not.
    """
    return any(isinstance(binding, Selectable) for binding in bindings.values())


DomainType = TypingUnion[Iterable[T], None]
