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

# Left side of hierarchy tree

PF_01 = SchematicNode(name='PF-01', is_load=True)
PF_02 = SchematicNode(name='PF-02', is_load=True)
PF_03 = SchematicNode(name='PF-03', is_load=True)
PF_05 = SchematicNode(name='PF-05', is_load=True)
PF_06 = SchematicNode(name='PF-06', is_load=True)
PF_07 = SchematicNode(name='PF-07', is_load=True)
PF_08 = SchematicNode(name='PF-08', is_load=True)

PF_201 = SchematicNode(name='PF-201', is_load=True)
PF_202 = SchematicNode(name='PF-202', is_load=True)
PF_203 = SchematicNode(name='PF-203', is_load=True)
PF_204 = SchematicNode(name='PF-204', is_load=True)
PF_205 = SchematicNode(name='PF-205', is_load=True)
PF_04 = SchematicNode(name='PF-04', is_load=True)

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

PF_09 = SchematicNode(name='PF-09', is_load=True)
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

print(strategy_variables)
outable_machines = [machine for machine in machines if not machine.is_load]
print(f"outable_machines : {outable_machines}")

for index, outable_machine in enumerate(outable_machines):
    outable_machine.strategy_variables = strategy_variables[index]

average_repair_times = [load.trace_average_repair_time() for load in load_machines]
print(f"average_repair_times : {average_repair_times}")

outage_costs = [random.randint(5000, 10000) for _ in range(len(load_machines))]
print(f"outage_costs : {outage_costs}")

loads = [random.randint(1, 10) * 1000 for _ in range(len(load_machines))]
print(f"loads : {loads}")

outage_rates = [machine.trace_outage_rate_variable() for machine in outable_machines]
print(f"outage_rates : {outage_rates}")

strategy_costs = [sorted([0.0] + [random.randint(1000, 5000) for _ in range(NUM_STRATEGY-1)]) for _ in range(
    NUM_OUTABLE_MACHINE)]
print(f"strategy_costs : {strategy_costs}")

num_users_of_loads = [random.randint(1, 10) * 1000 for _ in range(len(load_machines))]
print(f"num_users_of_loads : {num_users_of_loads}")

make_cost_constraint(scip_solver, strategy_variables, strategy_costs, 50000)
outage_rate_variables_of_loads = [machine.get_outage_rate_variable() for machine in outable_machines]
mean_repair_times = [_load.trace_average_repair_time() for _load in load_machines]

CIC_WEIGHT = 1/3.0
ENS_WEIGHT = 1/3.0
SAIFI_WEIGHT = 1/3.0

cic = sum([average_repair_times[i] * outage_costs[i] * loads[i] * outage_rate_variables_of_loads[i] for i in range(len(load_machines))])
ens = sum([loads[i] * outage_rate_variables_of_loads[i] * average_repair_times[i] for i in range(len(load_machines))])
saifi = sum([num_users_of_loads[i] * outage_rate_variables_of_loads[i] for i in range(len(load_machines))]) / sum(num_users_of_loads)

print(f"cic : {cic}")
print(f"ens : {ens}")
print(f"saifi : {saifi}")

scip_solver.Minimize(CIC_WEIGHT * cic + ENS_WEIGHT * ens + SAIFI_WEIGHT * saifi)

status = scip_solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print(f'Objective value = {scip_solver.Objective().Value()}')
    for index, machine in enumerate(outable_machines):
        print(f'{machine.name} : ', end='')
        for strategy in range(NUM_STRATEGY):
            print(f'{strategy_variables[index][strategy].solution_value()} ', end='')
else:
    print('The problem does not have an optimal solution.')
