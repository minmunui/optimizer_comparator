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


class FacilityNode:
    num_facility = 0

    def __init__(self,
                 name: str = None,
                 strategy_variables: list[Variable] = None,
                 outage_rates: list[int] = None,
                 average_outage_time: int = None,
                 is_load: bool = False):
        if len(outage_rates) == NUM_STRATEGY and len(strategy_variables) == NUM_STRATEGY:
            raise ValueError(f"Length of outage_rates and strategy_variables should be the same as NUM_STRATEGY"
                             f"NUM_STRATEGY : ({NUM_STRATEGY})"
                             f"Length of outage_rates : ({len(outage_rates)})"
                             f"Length of strategy_variables : ({len(strategy_variables)})")

        FacilityNode.num_facility += 1
        self.id = FacilityNode.num_facility
        self.name = name if name is not None else f'Facility_{FacilityNode.num_facility}'
        self.outage_rates = outage_rates
        self.average_outage_time = average_outage_time
        self.is_load = is_load
        self.strategy_variables = strategy_variables

    def __repr__(self):
        return f'FacilityNode({self.outage_rates}, {self.average_outage_time}, {self.is_load})'

    def __str__(self):
        return f'FacilityNode({self.outage_rates}, {self.average_outage_time}, {self.is_load})'

    def get_outage_rate_variable(self):
        return sum(
            [outage_rate * variable for outage_rate, variable in zip(self.outage_rates, self.strategy_variables)])


class SchematicNode:
    def __init__(self, value: FacilityNode, children: list['SchematicNode'] = None, parent: 'SchematicNode' = None):
        self.value = value
        self.children = children if children is not None else []
        self.parent = parent

    def __repr__(self):
        return f'SchematicNode({self.value})'

    def add_child(self, child: 'SchematicNode' | list['SchematicNode']):
        if isinstance(child, list):
            self.children.extend(child)
            for _child in child:
                _child.parent = self
        elif isinstance(child, SchematicNode):
            self.children.append(child)

        else :
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
        return sum([facility.value.outage_rates for facility in self.trace_root() if not facility.value.is_load])

    def trace_outage_rate_variable(self):
        linked_facilities = [facility for facility in self.trace_root() if not facility.value.is_load]
        outage_rate_variable = sum([facility.value.get_outage_rate_variable() for facility in linked_facilities])
        return outage_rate_variable

    def trace_average_repair_time(self) -> float:
        average_repair_time = [facility.value.average_outage_time for facility in self.trace_root() if
                               not facility.value.is_load]
        if len(average_repair_time) == 0:
            return 0
        return sum(average_repair_time) / len(average_repair_time)

    def remove_child(self, child):
        self.children.remove(child)


def make_cost_constraint(solver: pywraplp.Solver,
                         strategy_variables: list[list[Variable]],
                         strategy_cost: list[list[int]],
                         max_cost: int):
    for _machine in range(NUM_MACHINE):
        solver.Add(sum(strategy_cost[_machine][_strategy] * strategy_variables[_machine][_strategy] for _strategy in
                       range(NUM_STRATEGY)) <= max_cost)


def make_strategy_constraint(solver: pywraplp.Solver) -> list[list[Variable]]:
    x = list([0] * NUM_STRATEGY for _ in range(NUM_MACHINE))  # list[ list[0, 0, 0, 0], list[0, 0, 0, 0], ... ]
    for _machine in range(NUM_STRATEGY):
        for _strategy in range(NUM_STRATEGY):
            x[_machine][_strategy] = solver.IntVar(0, 1,
                                                   f'x_{_machine}_{_strategy}')  # each _strategy is binary variable

    for _machine in range(NUM_MACHINE):
        solver.Add(sum(x[_machine][_strategy] for _strategy in range(NUM_STRATEGY)) == 1)

    return x


if __name__ == '__main__':
    # Initialize the solver
    scip_solver = pywraplp.Solver.CreateSolver('SCIP')

    # Define the variables
    strategy_variables = make_strategy_constraint(solver=scip_solver)

    mat_outage_rate = [sorted([random.randint(1, 100) / 10000.0 for _ in range(NUM_STRATEGY)], reverse=True) for _ in
                       range(NUM_MACHINE)]
    load = [random.randint(10, 20) for _ in range(NUM_LOAD)]
    outage_cost = [random.randint(5000, 10000) for _ in range(NUM_LOAD)]
    num_user = [random.randint(1, 10) * 1000 for _ in range(NUM_LOAD)]

    strategy_cost = [sorted([random.randint(1000, 5000) for _ in range(NUM_STRATEGY)], reverse=True) for _ in
                     range(NUM_MACHINE)]

    print(f"mat_outage_rate : {mat_outage_rate}")
    print(f"load : {load}")
    print(f"outage_cost : {outage_cost}")
    print(f"num_user : {num_user}")
    print(f"strategy_cost : {strategy_cost}")

    # Define the constraints
    make_cost_constraint(scip_solver, strategy_variables, strategy_cost, 5000)

    # TODO : 설비 노드를 만들어서 계층 구조로 만들어야 함
    load_facilities = []

    # Outage rate of loads made by variables
    outage_rate_variables_of_loads = [_load.trace_outage_rate_variable() for _load in load_facilities]
    mean_repair_times = [_load.trace_average_repair_time() for _load in load_facilities]

    cic = sum(
        [mean_repair_times[i] * outage_cost[i] * load[i] * outage_rate_variables_of_loads[i] for i in range(NUM_LOAD)])
    ens = sum([load[i] * outage_rate_variables_of_loads[i] * mean_repair_times[i] for i in range(NUM_LOAD)])
    saifi = sum([num_user[i] * outage_rate_variables_of_loads[i] for i in range(NUM_LOAD)]) / sum(num_user)

    scip_solver.Maximize(WEIGHT_FOR_CIC * cic + WEIGHT_FOR_ENS * ens + WEIGHT_FOR_SAIFI * saifi)

    # Solve the problem
    status = scip_solver.Solve()

    # Print the results
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print(f'Objective value = {scip_solver.Objective().Value()}')
        for machine in range(NUM_MACHINE):
            for strategy in range(NUM_STRATEGY):
                print(f'x_{machine}_{strategy} = {strategy_variables[machine][strategy].solution_value()}')
    else:
        print('The problem does not have an optimal solution.')
