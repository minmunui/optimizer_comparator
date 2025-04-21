import random
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

random.seed(41)


class MachineNode:
    id = 0

    def __init__(self,
                 name: str = None,
                 system: 'SystemDiagram' = None,
                 parent: 'MachineNode' = None,
                 children: list['MachineNode'] = None,
                 strategy_costs: list[float] = None,
                 outage_rates: list[float] = None,
                 average_outage_time: float = None,
                 num_user: int = None,
                 outage_cost: float = None,
                 load: int = None,
                 ):
        self.system = system
        self.id = MachineNode.id
        self.name = name if name is not None else f"Node{self.id}"
        MachineNode.id += 1
        self.strategy_variables = []

        self.num_user = num_user
        self.outage_cost = outage_cost
        self.load = load

        self.children = []
        if children is not None:
            self.add_children(children)

        if parent is not None:
            self.parent = parent
            parent.children.append(self)
        else:
            self.parent = None

        self.strategy_costs = strategy_costs
        self.outage_rates = outage_rates
        self.average_outage_time = average_outage_time

    def __str__(self):
        return f"SchematicNode({self.name}, {self.num_user}, {self.outage_cost}, {self.load}, {self.outage_rates}, {self.strategy_costs}, {self.average_outage_time}, {[child.name for child in self.children]}, {self.parent.name if self.parent is not None else None})"

    def __repr__(self):
        return self.name

    def __format__(self, format_spec):
        return self.__str__()

    def add_child(self, child: 'MachineNode'):
        self.children.append(child)
        if self.system is not None:
            self.system.add_node(child)
        child.parent = self
        return child

    def remove_child(self, child: 'MachineNode'):
        self.children.remove(child)
        child.parent = None
        self.system.remove_node(child)
        return child

    def add_children(self, children: list['MachineNode']):
        for child in children:
            self.add_child(child)
        return children

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def trace_root(self):
        if self.is_root():
            return [self]
        return self.parent.trace_root() + [self]

    def get_root(self):
        return self.trace_root()[0]

    def is_outable(self):
        return self.outage_rates is not None and self.strategy_costs is not None

    def is_load(self):
        return self.num_user is not None and self.outage_cost is not None and self.load is not None

    def get_descendant_loads(self):
        if self.is_load():
            return [self]
        return sum([child.get_descendant_loads() for child in self.children], [])

    def get_cic_coef(self):
        if self.is_load():
            return self.outage_cost * self.load * self.get_average_outage_time()
        else:
            raise ValueError(f"Node {self.name} is not a load machine")

    def get_ens_coef(self):
        if self.is_load():
            return self.load * self.get_average_outage_time()
        else:
            raise ValueError(f"Node {self.name} is not a load machine")

    def get_saifi_coef(self):
        if self.is_load():
            return self.num_user
        else:
            raise ValueError(f"Node {self.name} is not a load machine")

    def find_child_loads(self):
        if self.is_outable():
            return sum([child.find_child_loads() for child in self.children], [])
        elif self.is_load():
            return [self]

    def get_cic_sensitivity(self):
        if self.is_outable():
            return [sum([child.get_cic_coef() for child in self.find_child_loads()]) * (
                    self.outage_rates[0] - outage_rate) / self.system.get_load_sum() for outage_rate in
                    self.outage_rates]
        else:
            raise ValueError(f"Node {self.name} is not an outable machine")

    def get_ens_sensitivity(self):
        if self.is_outable():
            return [
                sum([child.get_ens_coef() for child in self.find_child_loads()]) * (self.outage_rates[0] - outage_rate)
                for outage_rate in self.outage_rates]
        else:
            raise ValueError(f"Node {self.name} is not an outable machine")

    def get_saifi_sensitivity(self):
        if self.is_outable():
            return [sum([child.get_saifi_coef() * (self.outage_rates[0] - outage_rate) for child in
                         self.find_child_loads()]) / self.system.get_user_sum() for outage_rate in self.outage_rates]
        else:
            raise ValueError(f"Node {self.name} is not an outable machine")

    def get_average_outage_time(self):
        if self.is_load:
            return sum(machine.average_outage_time for machine in self.trace_root() if machine.is_outable()) / (len(
                self.trace_root()) - 1)
        raise ValueError(f"Node {self.name} is not a load machine")


class SystemDiagram:
    machines = []
    outable_machines = []
    load_machines = []

    def __init__(self,
                 name: str = None,
                 root: MachineNode = None,
                 ):
        self.name = name if name is not None else "System"
        self.root = root
        self.machines = []
        self.outable_machines = []
        self.load_machines = []
        if root is not None:
            self.add_node(root)

    def add_node(self, node: MachineNode):
        self.machines.append(node)
        node.system = self
        if node.num_user is not None or node.outage_cost is not None or node.load is not None:
            if node.num_user is None or node.outage_cost is None or node.load is None:
                raise ValueError(f"Load machine should have num_user, outage_cost, and load"
                                 f"but got {node.num_user}, {node.outage_cost}, {node.load}")
            # print(f"Node {node.name} is a load machine")
            self.load_machines.append(node)

        if node.outage_rates is not None and node.strategy_costs is not None:
            #             print(f"Node {node.name} is outable")
            self.outable_machines.append(node)

        for child in node.children:
            self.add_node(child)
        return node

    def remove_node(self, node: MachineNode):
        self.machines.remove(node)
        if node in self.load_machines:
            self.load_machines.remove(node)
        if node in self.outable_machines:
            self.outable_machines.remove(node)
        return node

    def set_root(self, root: MachineNode):
        self.root = root
        for machine in self.machines:
            self.remove_node(machine)
        self.add_node(root)
        return root

    def get_root(self):
        return self.root

    def print_tree(self, node=None, depth: int = 0):
        node = self.root if node is None else node
        print(' ' * depth, node.name)
        for child in node.children:
            self.print_tree(child, depth + 1)

    def make_strategy_cost_matrix(self):
        outables = sorted(self.outable_machines, key=lambda x: x.id)
        return [machine.strategy_costs for machine in outables]

    def get_load_sum(self):
        return sum([machine.load for machine in self.load_machines])

    def get_user_sum(self):
        return sum([machine.num_user for machine in self.load_machines])

    def get_current_saifi(self):
        return sum([machine.get_saifi_coef() for machine in self.load_machines])

    def make_cic_sensitivity_matrix(self):
        return [machine.get_cic_sensitivity() for machine in self.outable_machines]

    def make_ens_sensitivity_matrix(self):
        return [machine.get_ens_sensitivity() for machine in self.outable_machines]

    def make_saifi_sensitivity_matrix(self):
        return [machine.get_saifi_sensitivity() for machine in self.outable_machines]

    def make_cost_matrix(self):
        return [machine.strategy_costs for machine in self.outable_machines]

    def optimize_maintenance(self, ignore_strategy_nothing: bool = False, cost_constraint: float = 0, w_saifi: float = 1,
                             w_cic: float = 1,
                             w_ens: float = 1,
                             print_interval: int = 12,
                             solver:str = 'SCIP'  # 'SCIP' or 'CP-SAT'
                             ):
        if solver == 'SCIP':
            cost, optimizer, sensitivity_cic, sensitivity_ens, sensitivity_saifi, strategy_variables = self.make_SCIP_problem(
                ignore_strategy_nothing)

            optimizer.Add(cost <= cost_constraint)

            optimizer.Maximize(w_cic * sensitivity_cic + w_ens * sensitivity_ens + w_saifi * sensitivity_saifi)

            time_start = time.time()
            status = optimizer.Solve()
            time_end = time.time()

            print(f"Execution time: {time_end - time_start} seconds")

            if status == pywraplp.Solver.OPTIMAL:
                self.print_solution(ignore_strategy_nothing, optimizer, print_interval, strategy_variables)
                return True
        else:
            model = cp_model.CpModel()
            strategy_variables = []
            for outable_machine in self.outable_machines:
                for index, strategy in enumerate(outable_machine.strategy_costs):
                    if strategy == 0 and ignore_strategy_nothing:
                        pass
                    else:
                        strategy_variable = model.NewIntVar(0, 1, f"{outable_machine.name}_{index}")
                        outable_machine.strategy_variables.append(strategy_variable)
                strategy_variables.append(outable_machine.strategy_variables)
            # Define strategy constraint
            for strategy in strategy_variables:
                if ignore_strategy_nothing:
                    model.Add(sum(strategy) <= 1)
                else:
                    model.Add(sum(strategy) == 1)
            # Define objective function
            mat_cic = np.array(self.make_cic_sensitivity_matrix())
            mat_ens = np.array(self.make_ens_sensitivity_matrix())
            mat_saifi = np.array(self.make_saifi_sensitivity_matrix())
            mat_cost = np.array(self.make_cost_matrix())

            arr_strategy = np.array(strategy_variables)

            sensitivity_cic = sum(np.sum(mat_cic * arr_strategy, axis=1))
            sensitivity_ens = sum(np.sum(mat_ens * arr_strategy, axis=1))
            sensitivity_saifi = sum(np.sum(mat_saifi * arr_strategy, axis=1))

            cost = sum(np.sum(mat_cost * arr_strategy, axis=1))

            model.Minimize(cost)

            time_start = time.time()
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            time_end = time.time()

            print(f"Execution time: {time_end - time_start} seconds")

            if status == cp_model.OPTIMAL:
                self.print_solution(ignore_strategy_nothing, solver, print_interval, strategy_variables)
                return True

    def optimize_cost(self, obj_saifi, obj_cic, obj_ens, ignore_strategy_nothing: bool = False, solver: str = 'SCIP'):
        if solver == 'SCIP':
            cost, optimizer, sensitivity_cic, sensitivity_ens, sensitivity_saifi, strategy_variables = self.make_SCIP_problem(
                ignore_strategy_nothing)

            optimizer.Add(sensitivity_cic >= obj_cic)
            optimizer.Add(sensitivity_ens >= obj_ens)
            optimizer.Add(sensitivity_saifi >= obj_saifi)

            optimizer.Minimize(cost)

            time_start = time.time()
            status = optimizer.Solve()
            time_end = time.time()

            print(f"Execution time: {time_end - time_start} seconds")

            if status == pywraplp.Solver.OPTIMAL:
                self.print_solution(ignore_strategy_nothing, optimizer, 12, strategy_variables)
                return True
            else:
                print("No solution found")
                return False
        else :
            model = cp_model.CpModel()
            strategy_variables = []
            for outable_machine in self.outable_machines:
                for index, strategy in enumerate(outable_machine.strategy_costs):
                    if strategy == 0 and ignore_strategy_nothing:
                        pass
                    else:
                        strategy_variable = model.NewIntVar(0, 1, f"{outable_machine.name}_{index}")
                        outable_machine.strategy_variables.append(strategy_variable)
                strategy_variables.append(outable_machine.strategy_variables)
            # Define strategy constraint
            for strategy in strategy_variables:
                if ignore_strategy_nothing:
                    model.Add(sum(strategy) <= 1)
                else:
                    model.Add(sum(strategy) == 1)
            # Define objective function
            mat_cic = np.array(self.make_cic_sensitivity_matrix())
            mat_ens = np.array(self.make_ens_sensitivity_matrix())
            mat_saifi = np.array(self.make_saifi_sensitivity_matrix())
            mat_cost = np.array(self.make_cost_matrix())

            arr_strategy = np.array(strategy_variables)

            sensitivity_cic = sum(np.sum(mat_cic * arr_strategy, axis=1))
            sensitivity_ens = sum(np.sum(mat_ens * arr_strategy, axis=1))
            sensitivity_saifi = sum(np.sum(mat_saifi * arr_strategy, axis=1))

            cost = sum(np.sum(mat_cost * arr_strategy, axis=1))

            model.Minimize(cost)

            time_start = time.time()
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            time_end = time.time()

            print(f"Execution time: {time_end - time_start} seconds")

            if status == cp_model.OPTIMAL:
                self.print_solution(ignore_strategy_nothing, solver, 12, strategy_variables)
                return True

    def make_SCIP_problem(self, ignore_strategy_nothing):
        optimizer = pywraplp.Solver.CreateSolver('SCIP')
        strategy_variables = []
        for outable_machine in self.outable_machines:
            for index, strategy in enumerate(outable_machine.strategy_costs):
                if strategy == 0 and ignore_strategy_nothing:
                    pass
                else:
                    strategy_variable = optimizer.IntVar(0, 1, f"{outable_machine.name}_{index}")
                    outable_machine.strategy_variables.append(strategy_variable)
            strategy_variables.append(outable_machine.strategy_variables)
        # Define strategy constraint
        for strategy in strategy_variables:
            if ignore_strategy_nothing:
                optimizer.Add(sum(strategy) <= 1)
            else:
                optimizer.Add(sum(strategy) == 1)
        # Define objective function
        if ignore_strategy_nothing:
            mat_cic = np.array(self.make_cic_sensitivity_matrix())[:, 1:]
            mat_ens = np.array(self.make_ens_sensitivity_matrix())[:, 1:]
            mat_saifi = np.array(self.make_saifi_sensitivity_matrix())[:, 1:]
            mat_cost = np.array(self.make_cost_matrix())[:, 1:]
        else:
            mat_cic = np.array(self.make_cic_sensitivity_matrix())
            mat_ens = np.array(self.make_ens_sensitivity_matrix())
            mat_saifi = np.array(self.make_saifi_sensitivity_matrix())
            mat_cost = np.array(self.make_cost_matrix())
        arr_strategy = np.array(strategy_variables)
        sensitivity_cic = sum(np.sum(mat_cic * arr_strategy, axis=1))
        sensitivity_ens = sum(np.sum(mat_ens * arr_strategy, axis=1))
        sensitivity_saifi = sum(np.sum(mat_saifi * arr_strategy, axis=1))
        cost = sum(np.sum(mat_cost * arr_strategy, axis=1))
        return cost, optimizer, sensitivity_cic, sensitivity_ens, sensitivity_saifi, strategy_variables

    def print_solution(self, ignore_strategy_nothing, optimizer, print_interval, strategy_variables):
        print('Solution:')
        print(f'Objective value = {optimizer.Objective().Value()}')
        # print machine name
        print(f"{'Outable Machines':<{print_interval * 4}}", end='')
        for machine in self.outable_machines:
            print(f"{machine.name:{print_interval * 4}}", end='')
        print()
        print(f"{'CIC Sensitivity':<{print_interval * 4}}", end='')
        for machine in self.outable_machines:
            for sensitivity in machine.get_cic_sensitivity():
                print(f"{sensitivity:<{print_interval}.5f}", end='')
        print()
        print(f"{'ENS Sensitivity':<{print_interval * 4}}", end='')
        for machine in self.outable_machines:
            for sensitivity in machine.get_ens_sensitivity():
                print(f"{sensitivity:<{print_interval}.5f}", end='')
        print()
        print(f"{'SAIFI Sensitivity':<{print_interval * 4}}", end='')
        for machine in self.outable_machines:
            for sensitivity in machine.get_saifi_sensitivity():
                print(f"{sensitivity:<{print_interval}.5f}", end='')
        print()
        print(f"{'Repair time':<{print_interval * 4}}", end='')
        for repair_time in [machine.average_outage_time for machine in self.outable_machines]:
            print(f"{repair_time:<{print_interval * 4}.0f}", end='')
        # print costs of strategies
        print()
        print(f"{'Costs':<{print_interval * 4}}", end='')
        for costs in self.make_strategy_cost_matrix():
            for cost in costs:
                print(f"{cost:<{print_interval}}", end='')
        print()
        print(f"\n{'Strategy':<{print_interval * 4}}", end='')
        for index, machine in enumerate(self.outable_machines):
            if ignore_strategy_nothing:
                print(f"{'-':<{print_interval}}", end='')
                for strategy in range(len(machine.strategy_costs) - 1):
                    print(f'{strategy_variables[index][strategy].solution_value():<{print_interval}}', end='')
            else:
                for strategy in range(len(machine.strategy_costs)):
                    print(f'{machine.strategy_variables[strategy].solution_value():<{print_interval}}', end='')
        print()
        print(f"{'Outage rate':<{print_interval * 4}}", end='')
        for index, machine in enumerate(self.outable_machines):
            for strategy in range(len(machine.strategy_costs)):
                print(f'{machine.outage_rates[strategy]:<{print_interval}.3f}', end='')
        print()
        print()
        print("Load machine info:")
        print(f"{'Load Machines':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.name:{print_interval}}", end='')
        print()
        print(f"{'Num User':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.num_user:<{print_interval}}", end='')
        print()
        print(f"{'Outage cost':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.outage_cost:<{print_interval}}", end='')
        print()
        print(f"{'Load':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.load:<{print_interval}}", end='')
        print()
        print(f"{'Average outage time':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.get_average_outage_time():<{print_interval}.3f}", end='')
        print("")
        print(f"{'CIC of loads':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.get_cic_coef():<{print_interval}.3f}", end='')
        print("")
        print(f"{'ENS of loads':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.get_ens_coef():<{print_interval}.3f}", end='')
        print("")
        print(f"{'SAIFI of loads':<{print_interval * 4}}", end='')
        for machine in self.load_machines:
            print(f"{machine.get_saifi_coef():<{print_interval}.3f}", end='')
        print("")

if __name__ == "__main__":
    NUM_LOADS = 14
    NUM_STRATEGIES = 4
    NUM_MACHINES = 51
    outage_costs = [10000 for _ in range(NUM_LOADS)]
    # print(f"outage_costs : {outage_costs}")
    loads = [1000 for _ in range(NUM_LOADS)]
    # print(f"loads : {loads}")
    num_user = [1000 for _ in range(NUM_LOADS)]
    # print(f"num_users_of_loads : {num_user}")

    outage_rates = [sorted([0.004, 0.003, 0.002, 0.001], reverse=True) for _ in
                    range(NUM_MACHINES - NUM_LOADS)]
    strategy_costs = [sorted([0.0] + [random.randint(1000, 5000) for _ in range(NUM_STRATEGIES - 1)]) for _ in
                      range(NUM_MACHINES - NUM_LOADS)]
    average_outage_times = [1 for _ in range(NUM_MACHINES - NUM_LOADS)]

    # Define a system diagram
    PF_01 = MachineNode(name="PF_01", num_user=num_user[0], outage_cost=outage_costs[0], load=loads[0])
    PF_02 = MachineNode(name="PF_02", num_user=num_user[1], outage_cost=outage_costs[1], load=loads[1])
    PF_03 = MachineNode(name="PF_03", num_user=num_user[2], outage_cost=outage_costs[2], load=loads[2])
    PF_05 = MachineNode(name="PF_05", num_user=num_user[3], outage_cost=outage_costs[3], load=loads[3])
    PF_06 = MachineNode(name="PF_06", num_user=num_user[4], outage_cost=outage_costs[4], load=loads[4])
    PF_07 = MachineNode(name="PF_07", num_user=num_user[5], outage_cost=outage_costs[5], load=loads[5])
    PF_08 = MachineNode(name="PF_08", num_user=num_user[6], outage_cost=outage_costs[6], load=loads[6])

    PF_201 = MachineNode(name="PF_201", num_user=num_user[7], outage_cost=outage_costs[7], load=loads[7])
    PF_202 = MachineNode(name="PF_202", num_user=num_user[8], outage_cost=outage_costs[8], load=loads[8])
    PF_203 = MachineNode(name="PF_203", num_user=num_user[9], outage_cost=outage_costs[9], load=loads[9])
    PF_204 = MachineNode(name="PF_204", num_user=num_user[10], outage_cost=outage_costs[10], load=loads[10])
    PF_205 = MachineNode(name="PF_205", num_user=num_user[11], outage_cost=outage_costs[11], load=loads[11])
    PF_04 = MachineNode(name="PF_04", num_user=num_user[12], outage_cost=outage_costs[12], load=loads[12])

    GCB_01 = MachineNode(name="GCB_01", children=[PF_01], outage_rates=outage_rates[0],
                         strategy_costs=strategy_costs[0], average_outage_time=average_outage_times[0])
    GCB_02 = MachineNode(name="GCB_02", children=[PF_02], outage_rates=outage_rates[1],
                         strategy_costs=strategy_costs[1], average_outage_time=average_outage_times[1])
    GCB_03 = MachineNode(name="GCB_03", children=[PF_03], outage_rates=outage_rates[2],
                         strategy_costs=strategy_costs[2], average_outage_time=average_outage_times[2])
    GCB_05 = MachineNode(name="GCB_05", children=[PF_05], outage_rates=outage_rates[3],
                         strategy_costs=strategy_costs[3], average_outage_time=average_outage_times[3])
    GCB_06 = MachineNode(name="GCB_06", children=[PF_06], outage_rates=outage_rates[4],
                         strategy_costs=strategy_costs[4], average_outage_time=average_outage_times[4])
    GCB_07 = MachineNode(name="GCB_07", children=[PF_07], outage_rates=outage_rates[5],
                         strategy_costs=strategy_costs[5], average_outage_time=average_outage_times[5])
    GCB_08 = MachineNode(name="GCB_08", children=[PF_08], outage_rates=outage_rates[6],
                         strategy_costs=strategy_costs[6], average_outage_time=average_outage_times[6])

    LDS_01 = MachineNode(name="LDS_01", children=[GCB_01], outage_rates=outage_rates[7],
                         strategy_costs=strategy_costs[7], average_outage_time=average_outage_times[7])
    LDS_02 = MachineNode(name="LDS_02", children=[GCB_02], outage_rates=outage_rates[8],
                         strategy_costs=strategy_costs[8], average_outage_time=average_outage_times[8])
    LDS_03 = MachineNode(name="LDS_03", children=[GCB_03], outage_rates=outage_rates[9],
                         strategy_costs=strategy_costs[9], average_outage_time=average_outage_times[9])
    LDS_05 = MachineNode(name="LDS_05", children=[GCB_05], outage_rates=outage_rates[10],
                         strategy_costs=strategy_costs[10], average_outage_time=average_outage_times[10])
    LDS_06 = MachineNode(name="LDS_06", children=[GCB_06], outage_rates=outage_rates[11],
                         strategy_costs=strategy_costs[11], average_outage_time=average_outage_times[11])
    LDS_07 = MachineNode(name="LDS_07", children=[GCB_07], outage_rates=outage_rates[12],
                         strategy_costs=strategy_costs[12], average_outage_time=average_outage_times[12])
    LDS_08 = MachineNode(name="LDS_08", children=[GCB_08], outage_rates=outage_rates[13],
                         strategy_costs=strategy_costs[13], average_outage_time=average_outage_times[13])

    BUS_A = MachineNode(name="BUS_A", children=[LDS_01, LDS_02, LDS_03, LDS_05, LDS_06, LDS_07, LDS_08],
                        outage_rates=outage_rates[14], strategy_costs=strategy_costs[14],
                        average_outage_time=average_outage_times[14])

    VCB_A_3 = MachineNode(name="VCB_A_3", children=[BUS_A], outage_rates=outage_rates[15],
                          strategy_costs=strategy_costs[15], average_outage_time=average_outage_times[15])

    VCB_A_2 = MachineNode(name="VCB_A_2", children=[VCB_A_3], outage_rates=outage_rates[16],
                          strategy_costs=strategy_costs[16], average_outage_time=average_outage_times[16])

    PF_09 = MachineNode(name="PF_09", num_user=num_user[13], outage_cost=outage_costs[13], load=loads[13])
    VCB_09 = MachineNode(name="VCB_09", children=[PF_09], outage_rates=outage_rates[17],
                         strategy_costs=strategy_costs[17], average_outage_time=average_outage_times[17])

    BUS_09 = MachineNode(name="BUS_09", children=[VCB_09, VCB_A_2], outage_rates=outage_rates[18],
                         strategy_costs=strategy_costs[18], average_outage_time=average_outage_times[18])

    VCB_A_1 = MachineNode(name="VCB_A_1", children=[BUS_09], outage_rates=outage_rates[19],
                          strategy_costs=strategy_costs[19], average_outage_time=average_outage_times[19])
    cable_head_A = MachineNode(name="Cable head_A", children=[VCB_A_1], outage_rates=outage_rates[20],
                               strategy_costs=strategy_costs[20], average_outage_time=average_outage_times[20])
    TR_A = MachineNode(name="Tr._A", children=[cable_head_A], outage_rates=outage_rates[21],
                       strategy_costs=strategy_costs[21], average_outage_time=average_outage_times[21])
    GIS_TR_A = MachineNode(name="GIS_Tr_A", children=[TR_A], outage_rates=outage_rates[22],
                           strategy_costs=strategy_costs[22], average_outage_time=average_outage_times[22])

    GCB_201 = MachineNode(name="GCB_201", children=[PF_201], outage_rates=outage_rates[23],
                          strategy_costs=strategy_costs[23], average_outage_time=average_outage_times[23])
    GCB_202 = MachineNode(name="GCB_202", children=[PF_202], outage_rates=outage_rates[24],
                          strategy_costs=strategy_costs[24], average_outage_time=average_outage_times[24])
    GCB_203 = MachineNode(name="GCB_203", children=[PF_203], outage_rates=outage_rates[25],
                          strategy_costs=strategy_costs[25], average_outage_time=average_outage_times[25])
    GCB_204 = MachineNode(name="GCB_204", children=[PF_204], outage_rates=outage_rates[26],
                          strategy_costs=strategy_costs[26], average_outage_time=average_outage_times[26])
    GCB_205 = MachineNode(name="GCB_205", children=[PF_205], outage_rates=outage_rates[27],
                          strategy_costs=strategy_costs[27], average_outage_time=average_outage_times[27])
    GCB_04 = MachineNode(name="GCB_04", children=[PF_04], outage_rates=outage_rates[28],
                         strategy_costs=strategy_costs[28], average_outage_time=average_outage_times[28])

    BUS_B = MachineNode(name="BUS_B", children=[GCB_201, GCB_202, GCB_203, GCB_204, GCB_205, GCB_04],
                        outage_rates=outage_rates[29], strategy_costs=strategy_costs[29],
                        average_outage_time=average_outage_times[29])

    GCB_B_2 = MachineNode(name="GCB_B_2", children=[BUS_B], outage_rates=outage_rates[30],
                          strategy_costs=strategy_costs[30], average_outage_time=average_outage_times[30])
    GCB_B_1 = MachineNode(name="GCB_B_1", children=[GCB_B_2], outage_rates=outage_rates[31],
                          strategy_costs=strategy_costs[31], average_outage_time=average_outage_times[31])
    TR_B = MachineNode(name="Tr._B", children=[GCB_B_1], outage_rates=outage_rates[32],
                       strategy_costs=strategy_costs[32], average_outage_time=average_outage_times[32])
    GIS_TR_B = MachineNode(name="GIS_Tr_B", children=[TR_B], outage_rates=outage_rates[33],
                           strategy_costs=strategy_costs[33], average_outage_time=average_outage_times[33])

    BUS_main = MachineNode(name="BUS_main", children=[GIS_TR_A, GIS_TR_B], outage_rates=outage_rates[34],
                           strategy_costs=strategy_costs[34], average_outage_time=average_outage_times[34])
    GIS_TL = MachineNode(name="GIS_TL", children=[BUS_main], outage_rates=outage_rates[35],
                         strategy_costs=strategy_costs[35], average_outage_time=average_outage_times[35])
    cable_head_main = MachineNode(name="Cable head_main", children=[GIS_TL], outage_rates=outage_rates[36],
                                  strategy_costs=strategy_costs[36], average_outage_time=average_outage_times[36])
    # Print the system diagram

    system = SystemDiagram(root=cable_head_main)
    # system.print_tree()

    # print(f"{system.outable_machines}")
    # print(f"{system.load_machines}")
    print("Strategy cost matrix")
    print(system.make_strategy_cost_matrix())
    print("CIC sensitivity matrix")
    print(system.make_cic_sensitivity_matrix())
    print("SAIFI sensitivity matrix")
    print(system.make_saifi_sensitivity_matrix())
    print("ENS sensitivity matrix")
    print(system.make_ens_sensitivity_matrix())


    system.optimize_maintenance( w_saifi=1, w_cic=1, w_ens=1, solver="CP-SAT")
    # system.optimize_cost(obj_saifi=0.001, obj_cic=100, obj_ens=40, solver='CP-SAT')
