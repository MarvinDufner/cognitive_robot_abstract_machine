---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Result Processors

Result processors in EQL are mappings that are applied to the results produced from a query or variable. Currently, there are two kinds of result processors:

- Aggregators: `count`, `sum`, `average`, `max`, and `min`.
- Result Quantifiers: `the`, `a/an`, etc. See the dedicated page for details: {doc}`result_quantifiers`.

All result processors are evaluatable: they return a query object that exposes `.evaluate()`.

```{note}
You can pass either a variable created with `variable(...)` directly, or wrap it with `entity(...)`. Both forms are supported by the aggregators demonstrated below.
```

## Setup

```{code-cell} ipython3
from dataclasses import dataclass
from typing_extensions import List

import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity_result_processors import a, an
from krrood.entity_query_language.entity import entity, variable, contains, set_of


@dataclass
class Body:
    name: str
    type: str
    height: int


@dataclass
class World:
    bodies: List[Body]


world = World([
    Body("Handle1", "Handle", 1),
    Body("Handle2", "Handle", 2),
    Body("Container1", "Container", 3),
    Body("Container2", "Container", 4),
    Body("Container3", "Container", 5),
])
```

## count

Count the number of results matching a predicate.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = eql.count(
    entity(
        body).where(
        contains(body.name, "Handle"),
    )
)

print(next(query.evaluate()))  # -> 2
```

You can also count over a variable directly (without `entity(...)`).

```{code-cell} ipython3
query = eql.count(variable(Body, domain=world.bodies))
print(next(query.evaluate()))  # -> 5
```

## sum

Sum numeric values from the results. You can provide a `key` function to extract the numeric value from the results.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

query = eql.sum(body, key=lambda b: b.height)
print(next(query.evaluate()))  # -> 15
```

If there are no results, `sum` returns `None` by default. You can specify a `default` value.

```{code-cell} ipython3
empty = variable(int, domain=[])
query = eql.sum(empty, default=0)
print(next(query.evaluate()))  # -> 0
```

## average

Compute the arithmetic mean of numeric values. Like `sum`, it supports `key` and `default`.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)
query = eql.average(body, key=lambda b: b.height)
print(next(query.evaluate()))  # -> 3.0
```

## max and min

Find the maximum or minimum value. These also support `key` and `default`.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

max_query = eql.max(body, key=lambda b: b.height)
min_query = eql.min(body, key=lambda b: b.height)

print(next(max_query.evaluate()))  # -> Body(name='Container3', type='Container', height=5)
print(next(min_query.evaluate()))  # -> Body(name='Handle1', type='Handle', height=1)
```

## Grouped Aggregations

Aggregators support a `.per(*variables)` method that allows performing aggregations per group defined by one or more variables.

```{code-cell} ipython3
body = variable(Body, domain=world.bodies)

# Count bodies per type
count_per_body_type = eql.count(body).per(body.type)
query = a(set_of(count_per_body_type, body.type))

# evaluate the query and print the results
results = query.evaluate()

for res in results:
    group_type = res[body.type]
    count_value = res[count_per_body_type]
    print(f"Type: {group_type}, Count: {count_value}")
```


