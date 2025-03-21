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


def get_cic(strategy: list[int]):
    return sum(
        [load.trace_average_repair_time() * load.outage_cost * load.load * load.get_outage_rate(strategy) for load
         in SchematicNode.loads])


def get_saifi(strategy: list[int]):
    return sum([load.get_outage_rate(strategy) * load.num_user for load in SchematicNode.loads]) / sum(
        [load.num_user for load in SchematicNode.loads])


def get_ens(strategy: list[int]):
    return sum(
        [load.load * load.get_outage_rate(strategy) * load.trace_average_repair_time() for load in SchematicNode.loads])


def get_objective_value(strategy: list[int], weight_cic=WEIGHT_FOR_CIC, weight_ens=WEIGHT_FOR_ENS,
                        weight_saifi=WEIGHT_FOR_SAIFI, max_cost=50000):
    total_cost = get_strategy_cost(strategy)
    if total_cost > max_cost:
        return -1_000_000_000
    return weight_cic * get_cic(strategy) + weight_ens * get_ens(strategy) + weight_saifi * get_saifi(strategy)


def get_strategy_cost(strategy: list[int]):
    return sum(
        [machine.strategy_costs[strategy[index]] for index, machine in enumerate(SchematicNode.outable_machines)])


class SchematicNode:
    num_facility = 0
    loads = []
    machines = []
    outable_machines = []

    def __init__(self,
                 name: str = None,
                 children: list['SchematicNode'] = None,
                 parent: 'SchematicNode' = None,
                 strategy_variables: list[Variable] = None,
                 strategy_costs: list[float] = None,
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
            SchematicNode.outable_machines.append(self)
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
        self.strategy_costs = strategy_costs

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

    def get_outage_rate(self, strategy: list[int]):
        dict_strategy = dict(zip([m.name for m in SchematicNode.machines if not m.is_load], strategy))
        upper_machines = [m for m in self.trace_root() if not m.is_load]
        return sum([m.outage_rates[dict_strategy[m.name]] for m in upper_machines])

    def remove_child(self, child):
        self.children.remove(child)

    def print_tree(self, depth=0):
        print(' ' * depth, self.name)
        for child in self.children:
            child.print_tree(depth + 1)


def make_cost_constraint(solver: pywraplp.Solver,
                         max_cost: int):
    strategy_costs = [machine.strategy_costs for machine in SchematicNode.outable_machines]
    strategy_variables = [machine.strategy_variables for machine in SchematicNode.outable_machines]
    flattened_cost = [cost for strategy in strategy_costs for cost in strategy]
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
