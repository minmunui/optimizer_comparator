import random

from ortools.linear_solver import pywraplp

from transformer_opt import SchematicNode, NUM_STRATEGY, make_strategy_constraint, make_cost_constraint

random.seed(42)


def get_random_outage_rate():
    return sorted([random.randint(1, 100) / 10000.0 for _ in range(NUM_STRATEGY)], reverse=True)


def get_random_outage_time():
    return random.randint(1, 100) * 10.0


def get_random_schematic_node(name: str = None, children: list[SchematicNode] = None):
    return SchematicNode(name=name, children=children, outage_rates=get_random_outage_rate(),
                         average_outage_time=get_random_outage_time())


scip_solver = pywraplp.Solver.CreateSolver('SCIP')

NUM_MACHINES = 51
NUM_STRATEGY = 4
NUM_OUTABLE_MACHINE = 37
NUM_LOADS = 14

# Left side of hierarchy tree

outage_costs = [random.randint(5000, 10000) for _ in range(NUM_LOADS)]
print(f"outage_costs : {outage_costs}")
loads = [random.randint(1, 10) * 1000 for _ in range(NUM_LOADS)]
print(f"loads : {loads}")
num_user = [random.randint(1, 10) * 1000 for _ in range(NUM_LOADS)]
print(f"num_users_of_loads : {num_user}")

PF_01 = SchematicNode(name='PF-01', num_user=num_user[0], outage_cost=outage_costs[0], load=loads[0])
PF_02 = SchematicNode(name='PF-02', num_user=num_user[1], outage_cost=outage_costs[1], load=loads[1])
PF_03 = SchematicNode(name='PF-03', num_user=num_user[2], outage_cost=outage_costs[2], load=loads[2])
PF_05 = SchematicNode(name='PF-05', num_user=num_user[3], outage_cost=outage_costs[3], load=loads[3])
PF_06 = SchematicNode(name='PF-06', num_user=num_user[4], outage_cost=outage_costs[4], load=loads[4])
PF_07 = SchematicNode(name='PF-07', num_user=num_user[5], outage_cost=outage_costs[5], load=loads[5])
PF_08 = SchematicNode(name='PF-08', num_user=num_user[6], outage_cost=outage_costs[6], load=loads[6])

PF_201 = SchematicNode(name='PF-201', num_user=num_user[7], outage_cost=outage_costs[7], load=loads[7])
PF_202 = SchematicNode(name='PF-202', num_user=num_user[8], outage_cost=outage_costs[8], load=loads[8])
PF_203 = SchematicNode(name='PF-203', num_user=num_user[9], outage_cost=outage_costs[9], load=loads[9])
PF_204 = SchematicNode(name='PF-204', num_user=num_user[10], outage_cost=outage_costs[10], load=loads[10])
PF_205 = SchematicNode(name='PF-205', num_user=num_user[11], outage_cost=outage_costs[11], load=loads[11])
PF_04 = SchematicNode(name='PF-04', num_user=num_user[12], outage_cost=outage_costs[12], load=loads[12])

# 37


GCB_01 = get_random_schematic_node(name='GCB_01', children=[PF_01])
GCB_02 = get_random_schematic_node(name='GCB_02', children=[PF_02])
GCB_03 = get_random_schematic_node(name='GCB_03', children=[PF_03])
GCB_05 = get_random_schematic_node(name='GCB_05', children=[PF_05])
GCB_06 = get_random_schematic_node(name='GCB_06', children=[PF_06])
GCB_07 = get_random_schematic_node(name='GCB_07', children=[PF_07])
GCB_08 = get_random_schematic_node(name='GCB_08', children=[PF_08])

LDS_01 = get_random_schematic_node(name='LDS_01', children=[GCB_01])
LDS_02 = get_random_schematic_node(name='LDS_02', children=[GCB_02])
LDS_03 = get_random_schematic_node(name='LDS_03', children=[GCB_03])
LDS_05 = get_random_schematic_node(name='LDS_05', children=[GCB_05])
LDS_06 = get_random_schematic_node(name='LDS_06', children=[GCB_06])
LDS_07 = get_random_schematic_node(name='LDS_07', children=[GCB_07])
LDS_08 = get_random_schematic_node(name='LDS_08', children=[GCB_08])

BUS_A = get_random_schematic_node(name='BUS_A', children=[LDS_01, LDS_02, LDS_03, LDS_05, LDS_06, LDS_07, LDS_08])

VCB_A_3 = get_random_schematic_node(name='VCB_A_3', children=[BUS_A])

VCB_A_2 = get_random_schematic_node(name='VCB_A_2', children=[VCB_A_3])

PF_09 = SchematicNode(name='PF-09', num_user=num_user[13], outage_cost=outage_costs[13], load=loads[13])
VCB_09 = get_random_schematic_node(name='VCB_09', children=[PF_09])

BUS_09 = get_random_schematic_node(name='BUS_09', children=[VCB_09, VCB_A_2])

VCB_A_1 = get_random_schematic_node(name='VCB_A_1', children=[BUS_09])
cable_head_A = get_random_schematic_node(name='Cable head_A', children=[VCB_A_1])
TR_A = get_random_schematic_node(name='Tr._A', children=[cable_head_A])
GIS_TR_A = get_random_schematic_node(name='GIS_Tr_A', children=[TR_A])

# Right side of hierarchy tree

GCB_201 = get_random_schematic_node(name='GCB_201', children=[PF_201])
GCB_202 = get_random_schematic_node(name='GCB_202', children=[PF_202])
GCB_203 = get_random_schematic_node(name='GCB_203', children=[PF_203])
GCB_204 = get_random_schematic_node(name='GCB_204', children=[PF_204])
GCB_205 = get_random_schematic_node(name='GCB_205', children=[PF_205])
GCB_04 = get_random_schematic_node(name='GC_B04', children=[PF_04])

BUS_B = get_random_schematic_node(name='BUS_B', children=[GCB_201, GCB_202, GCB_203, GCB_204, GCB_205, GCB_04])

GCB_B_2 = get_random_schematic_node(name='GCB_B_2', children=[BUS_B])
GCB_B_1 = get_random_schematic_node(name='GCB_B_1', children=[GCB_B_2])
TR_B = get_random_schematic_node(name='Tr._B', children=[GCB_B_1])
GIS_TR_B = get_random_schematic_node(name='GIS_Tr_B', children=[TR_B])

# Main hierarchy tree

BUS_main = get_random_schematic_node(name='BUS_main', children=[GIS_TR_A, GIS_TR_A])
GIS_TL = get_random_schematic_node(name='GIS_TL', children=[BUS_main])
cable_head_main = get_random_schematic_node(name='Cable head_main', children=[GIS_TL])

load_machines = SchematicNode.loads
machines = SchematicNode.machines

cable_head_main.print_tree()

strategy_variables = make_strategy_constraint(solver=scip_solver, num_machine=NUM_OUTABLE_MACHINE,
                                              num_strategy=NUM_STRATEGY)

outable_machines = [machine for machine in machines if not machine.is_load]

for index, outable_machine in enumerate(outable_machines):
    outable_machine.strategy_variables = strategy_variables[index]

average_repair_times = [load.trace_average_repair_time() for load in load_machines]

outage_rates = [machine.trace_outage_rate_variable() for machine in outable_machines]
strategy_costs = [sorted([0.0] + [random.randint(1000, 5000) for _ in range(NUM_STRATEGY-1)]) for _ in range(
    NUM_OUTABLE_MACHINE)]



make_cost_constraint(scip_solver, strategy_variables, strategy_costs, 50000)
outage_rate_variables_of_loads = [machine.get_outage_rate_variable() for machine in outable_machines]
mean_repair_times = [_load.trace_average_repair_time() for _load in load_machines]

CIC_WEIGHT = 1/3.0
ENS_WEIGHT = 1/3.0
SAIFI_WEIGHT = 1/3.0

cic = sum([load.trace_average_repair_time() * load.outage_cost * load.load * load.trace_outage_rate_variable() for load in load_machines])
ens = sum([load.load * load.trace_outage_rate_variable() * load.trace_average_repair_time() for load in load_machines])
saifi = sum([load.trace_outage_rate_variable() * load.num_user for load in load_machines]) / sum([load.num_user for load in load_machines])


scip_solver.Minimize(CIC_WEIGHT * cic + ENS_WEIGHT * ens + SAIFI_WEIGHT * saifi)

status = scip_solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print(f'Objective value = {scip_solver.Objective().Value()}')

    # print machine name
    for machine in outable_machines:
        print(f"{machine.name:24}", end='')
    print()

    for repair_time in [machine.average_outage_time for machine in outable_machines]:
        print(f"{repair_time:<24.0f}", end='')

    # print costs of strategies
    for costs in strategy_costs:
        for cost in costs:
            print(f"{cost:<6}", end='')
    print()

    for index, machine in enumerate(outable_machines):
        for strategy in range(NUM_STRATEGY):
            print(f'{strategy_variables[index][strategy].solution_value():<6}', end='')

    print()

    for index, machine in enumerate(outable_machines):
        for strategy in range(NUM_STRATEGY):
            print(f'{machine.outage_rates[strategy]:<6.3f}', end='')

else:
    print('The problem does not have an optimal solution.')
