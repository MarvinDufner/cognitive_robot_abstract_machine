from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from typing_extensions import (
    Any,
    List,
    TypeVar,
    Generic,
    Type,
    TYPE_CHECKING,
    Dict,
    Optional,
    Generator,
)

from krrood.entity_query_language.entity import variable
from krrood.entity_query_language.symbolic import Variable
from .datastructures.dataclasses import Context

if TYPE_CHECKING:
    from .plan import Plan
    from .datastructures.partial_designator import PartialDesignator

logger = logging.getLogger("pycram")

T = TypeVar("T")


@dataclass
class ParameterInferer:
    """
    Central module to manage the inference of domains and parameter.

    Principle:
        Domains define general space of values for a type and are defined per type
        Rules: restrict the domain and are defined for certain parameters
    """

    parameter_rules: List[InferenceRule] = field(init=False, default_factory=list)
    """
    A set of rules that restrict the domain
    """

    inference_systems: List[InferenceSystem] = field(init=False, default_factory=list)

    plan_domain: PlanDomain = field(init=False)

    plan: Plan = None
    """
    Back-reference to the plan to which this parameterizer belongs 
    """

    def __post_init__(self):
        self.plan_domain = PlanDomain(self.plan)

    def add_rule(self, inference_rule: InferenceRule):
        """
        Adds a rule to the set of restriction rules

        :param inference_rule: The rule to add
        """
        self.parameter_rules.append(inference_rule)
        inference_rule.parameter_infeerer = self

    def add_domain(self, domain: DomainSpecification[Type[T]]):
        """
        Adds a domain to the set of domains

        :param domain: The domain to add
        """
        self.plan_domain.add_domain(domain)
        self.plan_domain.create_plan_domain()

    def add_infer_system(self, sys: InferenceSystem):
        self.inference_systems.append(sys)
        sys.assign_parameterizer(self)

    def add_domains(self, *domains: DomainSpecification[Type[T]]):
        """
        Adds a set set of domains
        """
        for domain in domains:
            self.add_domain(domain)

    def parameterize(self, description: PartialDesignator):
        return self.inference_systems[0].infer_bindings_for_designator(description)


@dataclass
class InferenceRule(Generic[T], ABC):
    """
    Rule that restricts a domain
    """

    parameter_infeerer: ParameterInferer = field(init=False)

    @abstractmethod
    def _apply(self, domain: List[T], context: Context) -> List[T]: ...

    def apply(self, domain: List[T], context: Context) -> List[T]:
        """
        Applies this rule to the domain
        """
        domain = self._apply(domain, context)
        self.effect()
        return domain

    def effect(self): ...

    """
    Effect that is triggered ones this rule fires. 
    """


@dataclass
class ParameterIdentifier:
    """
    Identifies a specific parameter of an action description
    """

    action_description: PartialDesignator
    """
    Description of an action to which the parameter belongs
    """

    parameter_name: str
    """
    Name of the parameter
    """

    @property
    def parameter(self) -> T:
        return self.action_description.kwargs[self.parameter_name]

    @property
    def type_(self) -> Type[T]:
        action_types = {
            f.name: f.type for f in self.action_description.performable.fields
        }
        return action_types[self.parameter_name]

    @property
    def path(self) -> str:
        return f"{self.action_description.performable}.{self.parameter_name}"

    def __hash__(self):
        return id(self)


@dataclass
class PlanDomain:
    plan: Plan

    designator_domains: Dict[PartialDesignator, DesignatorDomain] = field(
        default_factory=dict, init=False
    )

    domain_specifications: List[DomainSpecification] = field(
        default_factory=list, init=False
    )

    def create_plan_domain(self):
        for description_node in self.plan.actions:
            domains = {
                f.name: self.find_domain(
                    ParameterIdentifier(description_node.designator_ref, f.name)
                )
                for f in description_node.designator_ref.performable.fields
            }
            self.designator_domains[description_node.designator_ref] = DesignatorDomain(
                description_node.designator_ref,
                description_node.kwargs,
                domains,
                self.plan,
            )

    def find_domain(self, parameter_identifier: ParameterIdentifier):
        parameter_type = parameter_identifier.type_
        specs = [
            spec
            for spec in self.domain_specifications
            if spec.domain_type == parameter_type
        ]
        if specs:
            return specs[0]
        elif parameter_identifier.parameter is not Ellipsis:
            return ValueDomainSpecification(
                parameter_type, parameter_identifier.parameter
            )
        else:
            return ValueDomainSpecification(None, [])

    def add_domain(self, domain: DomainSpecification):
        self.domain_specifications.append(domain)

    def add_domains(self, *domains: DomainSpecification):
        for domain in domains:
            self.add_domain(domain)

    def get_domain_for_type(self, type_) -> Optional[DomainSpecification]:
        for domain in self.domain_specifications:
            if domain.domain_type == type_:
                return domain
        return None


@dataclass
class DesignatorDomain:
    designator: PartialDesignator

    kwargs: Dict[str, Any]

    parameter_domains: Dict[str, DomainSpecification]

    plan: Plan = None

    def domain(self) -> List:
        d = []
        for designator_domain in self.parameter_domains.values():
            d.extend(designator_domain.domain(self.plan.context))
        return d


@dataclass
class DomainSpecification(Generic[T], ABC):
    """
    Defines a domain of values. A domain are all possible values that a type can be.
    """

    domain_type: Type[T]
    """
    The type over which the domain is defined
    """

    @abstractmethod
    def domain(self, context: Context) -> List[T]: ...


@dataclass
class ValueDomainSpecification(DomainSpecification):

    domain_values: List[T]

    def domain(self, context: Context) -> List[T]:
        return [self.domain_values]


@dataclass
class InferenceSystem(ABC):

    plan: Plan = field(init=False)

    plan_domain: PlanDomain = field(init=False)

    @abstractmethod
    def infer_bindings_for_designator(
        self, designator: PartialDesignator
    ) -> Generator[Dict[str, Any]]:
        pass

    def create_variables(
        self, designator_description: PartialDesignator
    ) -> Dict[str, Variable]:
        designator_domain = self.plan_domain.designator_domains[designator_description]
        return {
            k: variable(v.domain_type, v.domain(self.plan.context))
            for k, v in designator_domain.parameter_domains.items()
        }

    def assign_parameterizer(self, parameterizer: ParameterInferer):
        self.plan_domain = parameterizer.plan_domain
        self.plan = parameterizer.plan
