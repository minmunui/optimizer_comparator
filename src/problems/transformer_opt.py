import json
from typing import Optional, Any

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


def get_init_cic():
    init_strategy = [0] * len(SchematicNode.outable_machines)
    return (sum(
        [load.trace_average_repair_time() * load.outage_cost * load.load * load.get_outage_rate(init_strategy) for
         load in SchematicNode.loads]))


def get_cic(strategy: list[int]):
    return sum(
        [load.trace_average_repair_time() * load.outage_cost * load.load * load.get_outage_rate(strategy) for load
         in SchematicNode.loads])


def get_cic_variables():
    return sum(
        [load.trace_average_repair_time() * load.outage_cost * load.load * load.trace_outage_rate_variable() for load in
         SchematicNode.loads])


def get_cic_sensitivity(strategy: list[int]):
    return get_init_cic() - get_cic(strategy)


def get_cic_sensitivity_variables():
    return get_init_cic() - get_cic_variables()


def get_init_saifi():
    init_strategy = [0] * len(SchematicNode.outable_machines)
    return sum([load.get_outage_rate(init_strategy) * load.num_user for load in SchematicNode.loads]) / sum(
        [load.num_user for load in SchematicNode.loads])


def get_saifi(strategy: list[int]):
    return sum([load.get_outage_rate(strategy) * load.num_user for load in SchematicNode.loads]) / sum(
        [load.num_user for load in SchematicNode.loads])


def get_saifi_variables():
    return sum([load.trace_outage_rate_variable() * load.num_user for load in SchematicNode.loads]) / sum(
        [load.num_user for load in SchematicNode.loads])


def get_saifi_sensitivity(strategy: list[int]):
    return get_init_saifi() - get_saifi(strategy)


def get_saifi_sensitivity_variables():
    return get_init_saifi() - get_saifi_variables()


def get_init_ens():
    init_strategy = [0] * len(SchematicNode.outable_machines)
    return sum([load.load * load.get_outage_rate(init_strategy) * load.trace_average_repair_time() for load in
                SchematicNode.loads])


def get_ens(strategy: list[int]):
    return sum(
        [load.load * load.get_outage_rate(strategy) * load.trace_average_repair_time() for load in SchematicNode.loads])


def get_ens_variables():
    return sum([load.load * load.trace_outage_rate_variable() * load.trace_average_repair_time() for load in
                SchematicNode.loads])


def get_ens_sensitivity(strategy: list[int]):
    return get_init_ens() - get_ens(strategy)


def get_ens_sensitivity_variables():
    return get_init_ens() - get_ens_variables()


def get_objective_reliability(strategy: list[int], weight_cic=WEIGHT_FOR_CIC, weight_ens=WEIGHT_FOR_ENS,
                              weight_saifi=WEIGHT_FOR_SAIFI, max_cost=50000, is_overcost_penalty=False):
    total_cost = get_strategy_cost(strategy)
    if total_cost > max_cost:
        if is_overcost_penalty:
            return (max_cost - total_cost) / 1_000_000_000
        return 0
    return weight_cic * get_cic_sensitivity(strategy) + weight_ens * get_ens_sensitivity(
        strategy) + weight_saifi * get_saifi_sensitivity(strategy)


def get_objective_cost(strategy: list[int]):
    return get_strategy_cost(strategy)


def get_strategy_cost(strategy: list[int]):
    return sum(
        [machine.strategy_costs[strategy[index]] for index, machine in enumerate(SchematicNode.outable_machines)])


def solution_to_strategy(solution: list[list[int]]):
    return [strategy.index(1) for strategy in solution]


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

    def get_root(self) -> 'SchematicNode':
        if self.is_root():
            return self
        return self.parent.get_root()

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

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, Any], parent: Optional['SchematicNode'] = None) -> 'SchematicNode':
        """JSON 딕셔너리에서 SchematicNode 객체 생성"""
        # 기본 속성 추출
        name = json_dict.get('name')

        # load 노드인 경우
        if json_dict.get('is_load', False):
            node = cls(
                name=name,
                parent=parent,
                load=json_dict.get('load'),
                num_user=json_dict.get('num_user'),
                outage_cost=json_dict.get('outage_cost')
            )
        # facility 노드인 경우
        else:
            # strategy_variables는 Variable 객체 리스트이므로 역직렬화가 어려울 수 있음
            # 여기서는 간단히 None으로 설정하고, 필요에 따라 나중에 설정하도록 함
            node = cls(
                name=name,
                parent=parent,
                outage_rates=json_dict.get('outage_rates'),
                average_outage_time=json_dict.get('average_outage_time'),
                strategy_costs=json_dict.get('strategy_costs'),
                strategy_variables=None  # 나중에 별도로 설정 필요
            )

        # 자식 노드 재귀적으로 생성
        children_data = json_dict.get('children', [])
        for child_data in children_data:
            cls.from_json_dict(child_data, parent=node)

        return node

    def to_json(self) -> dict[str, Any]:
        """현재 노드와 모든 하위 노드를 JSON으로 직렬화"""
        node_dict = {
            'name': self.name,
            'id': self.id,
            'is_load': self.is_load
        }

        # load 관련 속성 추가
        if self.is_load:
            node_dict.update({
                'load': self.load,
                'num_user': self.num_user,
                'outage_cost': self.outage_cost
            })
        # facility 관련 속성 추가
        else:
            if self.outage_rates is not None:
                node_dict['outage_rates'] = self.outage_rates
            if self.average_outage_time is not None:
                node_dict['average_outage_time'] = self.average_outage_time
            if self.strategy_costs is not None:
                node_dict['strategy_costs'] = self.strategy_costs

            # strategy_variables는 Variable 객체 리스트이므로 직렬화가 어려울 수 있음
            # 실제 구현에서는 Variable 클래스의 직렬화 방법에 따라 조정 필요
            if self.strategy_variables is not None:
                node_dict['strategy_variables_count'] = len(self.strategy_variables)

        node_dict['children'] = [child.to_json() for child in self.children]

        return node_dict

    def to_json_string(self, indent=2) -> str:
        """트리 구조를 JSON 문자열로 변환"""
        return json.dumps(self.to_json(), indent=indent)

    def save_to_json_file(self, filename: str) -> None:
        """트리 구조를 JSON 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def from_json_string(cls, json_string: str) -> 'SchematicNode':
        """JSON 문자열에서 트리 구조 로드"""
        json_dict = json.loads(json_string)
        return cls.from_json_dict(json_dict)

    @classmethod
    def from_json_file(cls, filename: str) -> 'SchematicNode':
        """JSON 파일에서 트리 구조 로드"""
        with open(filename, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
        return cls.from_json_dict(json_dict)

    @classmethod
    def reset_class_variables(cls):
        """클래스 변수 초기화 (새로운 트리 로드 시 사용)"""
        cls.num_facility = 0
        cls.loads = []
        cls.machines = []
        cls.outable_machines = []


def cost_variables():
    strategy_costs = [machine.strategy_costs for machine in SchematicNode.outable_machines]
    strategy_variables = [machine.strategy_variables for machine in SchematicNode.outable_machines]
    flattened_cost = [cost for strategy in strategy_costs for cost in strategy]
    flattened_variables = [variable for strategy in strategy_variables for variable in strategy]
    return sum([cost * variable for cost, variable in zip(flattened_cost, flattened_variables)])


def make_cost_constraint(solver: pywraplp.Solver, max_cost: int):
    solver.Add(cost_variables() <= max_cost)


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


def apply_reliability_objectives(solver: pywraplp.Solver,
                                 cic_weight: float = 1 / 3.0,
                                 ens_weight: float = 1 / 3.0,
                                 saifi_weight: float = 1 / 3.0
                                 ):
    solver.Maximize(
        cic_weight * get_cic_sensitivity_variables() +
        ens_weight * get_ens_sensitivity_variables() +
        saifi_weight * get_saifi_sensitivity_variables()
    )
