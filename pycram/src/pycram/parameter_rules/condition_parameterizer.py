from typing import Dict, Any

from typing_extensions import Generator

from krrood.entity_query_language.entity import set_of
from krrood.entity_query_language.entity_result_processors import a
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.parameter_inference import InferenceSystem


class ConditionParameterizer(InferenceSystem):

    def infer_bindings_for_designator(
        self, designator: PartialDesignator
    ) -> Generator[Dict[str, Any]]:
        variables = self.create_variables(designator)

        unbound_condition = designator.performable.pre_condition(
            variables, self.plan.context, designator.kwargs
        )

        query = a(set_of(*variables.values()).where(unbound_condition))
        var_to_field = dict(zip(variables.values(), designator.performable.fields))
        for result in query.evaluate():
            bindings = result.data
            yield {var_to_field[k].name: v for k, v in bindings.items()}
