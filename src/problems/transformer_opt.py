import random

from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable

LOAD = []
OUTAGE_COST = []
NUM_USER = []

NUM_MACHINE = 50
NUM_LOAD = 10
NUM_STRATEGY = 4

WEIGHT_FOR_CIC = 1.0 / 3.0
WEIGHT_FOR_ENS = 1.0 / 3.0
WEIGHT_FOR_SAIFI = 1.0 / 3.0


class SchematicNode:
    num_facility = 0
    loads = []
    machines = []

    def __init__(self,
                 name: str = None,
                 children: list['SchematicNode'] = None,
                 parent: 'SchematicNode' = None,
                 strategy_variables: list[Variable] = None,
                 outage_rates: list[float] = None,
                 average_outage_time: float = None,
                 num_user: int = None,
                 outage_cost: int = None,
                 load: float = None,
                 ):

        if children is None:
            self.children = []
        elif isinstance(children, list):
            self.children = children
            for child in children:
                child.parent = self

        if parent is not None:
            parent.add_child(self)
        self.parent = parent

        if load and num_user and outage_cost:
            self.load = load
            self.num_user = num_user
            self.outage_cost = outage_cost
            self.is_load = True
            SchematicNode.loads.append(self)
        elif load is None and num_user is None and outage_cost is None:
            self.is_load = False
        else:
            raise ValueError(f"load, num_user, outage_cost should be all None or all not None")

        self.num_user = num_user
        self.outage_cost = outage_cost

        SchematicNode.machines.append(self)
        SchematicNode.num_facility += 1
        self.id = SchematicNode.num_facility
        self.name = name if name is not None else f'Facility_{SchematicNode.num_facility}'

        self.outage_rates = outage_rates
        self.average_outage_time = average_outage_time
        self.strategy_variables = strategy_variables

    def add_child(self, child):
        if isinstance(child, list):
            self.children.extend(child)
            for _child in child:
                _child.parent = self
        elif isinstance(child, SchematicNode):
            self.children.append(child)

        else:
            raise ValueError(f"child should be SchematicNode or list[SchematicNode]")
        child.parent = self

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def trace_root(self) -> list['SchematicNode']:
        if self.is_root():
            return [self]
        return self.parent.trace_root() + [self]

    def trace_outage_rate(self) -> float:
        return sum([facility.outage_rates for facility in self.trace_root() if not facility.is_load])

    def trace_outage_rate_variable(self):
        linked_facilities = [facility for facility in self.trace_root() if not facility.is_load]
        outage_rate_variable = sum([facility.get_outage_rate_variable() for facility in linked_facilities])
        return outage_rate_variable

    def trace_average_repair_time(self) -> float:
        average_repair_time = [facility.average_outage_time for facility in self.trace_root() if
                               not facility.is_load]
        if len(average_repair_time) == 0:
            return 0
        return sum(average_repair_time) / len(average_repair_time)

    def get_outage_rate_variable(self):
        return sum(
            [outage_rate * variable for outage_rate, variable in zip(self.outage_rates, self.strategy_variables)])

    def remove_child(self, child):
        self.children.remove(child)

    def print_tree(self, depth=0):
        print(' ' * depth, self.name)
        for child in self.children:
            child.print_tree(depth + 1)


def make_cost_constraint(solver: pywraplp.Solver,
                         strategy_variables: list[list[Variable]],
                         strategy_cost: list[list[float]],
                         max_cost: int):
    flattened_cost = [cost for strategy in strategy_cost for cost in strategy]
    flattened_variables = [variable for strategy in strategy_variables for variable in strategy]
    solver.Add(sum([cost * variable for cost, variable in zip(flattened_cost, flattened_variables)]) <= max_cost)


def make_strategy_constraint(solver: pywraplp.Solver,
                             num_strategy: int = NUM_STRATEGY,
                             num_machine: int = NUM_MACHINE) \
        -> list[list[Variable]]:
    x = list([0] * num_strategy for _ in range(num_machine))  # list[ list[0, 0, 0, 0], list[0, 0, 0, 0], ... ]
    for _machine in range(num_machine):
        for _strategy in range(num_strategy):
            x[_machine][_strategy] = solver.IntVar(0, 1,
                                                   f'x_{_machine}_{_strategy}')  # each _strategy is binary variable

    for _machine in range(num_machine):
        solver.Add(sum(x[_machine][_strategy] for _strategy in range(num_strategy)) == 1)

    return x
