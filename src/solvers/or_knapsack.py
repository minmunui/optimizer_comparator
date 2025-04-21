import time

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


def solve_fractional_knapsack_or(values: list[int],
                                 weights: list[int],
                                 capacity: int,
                                 fraction: int,
                                 solver_name: str = 'SCIP',
                                 num_workers: int = 1):
    n = len(values)

    if solver_name == 'SCIP':
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("SCIP 솔버를 생성할 수 없습니다.")
            return None

        x = [solver.IntVar(0, fraction, f'x{i}') for i in range(n)]
        solver.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity * fraction)

        objective = solver.Objective()
        for i in range(n):
            objective.SetCoefficient(x[i], values[i])
        objective.SetMaximization()

        time_start = time.time()
        status = solver.Solve()
        time_end = time.time()
        print(f"[SCIP] Solver time: {time_end - time_start:.4f} seconds")

        if status == pywraplp.Solver.OPTIMAL:
            total_value = sum(values[i] * x[i].solution_value() for i in range(n)) / fraction
            total_weight = sum(weights[i] * x[i].solution_value() for i in range(n)) / fraction
            solution = [x[i].solution_value() for i in range(n)]
            return total_value, solution, total_weight, None, None
        else:
            print("최적 해를 찾지 못했습니다.")
            return None

    elif solver_name == 'CP-SAT':
        model = cp_model.CpModel()
        x = [model.NewIntVar(0, fraction, f'x{i}') for i in range(n)]

        model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity * fraction)
        model.Maximize(sum(values[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = num_workers
        # model.ExportToFile("model.txt")  # 필요 시 주석 해제

        time_start = time.time()
        status = solver.Solve(model)
        time_end = time.time()
        print(f"[CP-SAT] Solver time: {time_end - time_start:.4f} seconds")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            total_value = sum(values[i] * solver.Value(x[i]) for i in range(n)) / fraction
            total_weight = sum(weights[i] * solver.Value(x[i]) for i in range(n)) / fraction
            solution = [solver.Value(x[i]) for i in range(n)]
            return total_value, solution, total_weight, solver.WallTime(), solver.NumConflicts()
        else:
            print("해를 찾을 수 없습니다.")
            return None
    else:
        print(f"알 수 없는 solver_name: {solver_name}")
        return None


# 예제 실행
if __name__ == "__main__":
    def generate_knapsack_problem(num_items, weight_range, value_range, capacity_ratio):
        import random
        weights = [random.randint(*weight_range) for _ in range(num_items)]
        values = [random.randint(*value_range) for _ in range(num_items)]
        capacity = int(sum(weights) * capacity_ratio)
        return weights, values, capacity

    weights, values, capacity = generate_knapsack_problem(
        num_items=1000,
        weight_range=(10, 50),
        value_range=(50, 200),
        capacity_ratio=0.5
    )

    # 기본 실행
    result = solve_fractional_knapsack_or(values, weights, capacity, 1, solver_name='CP-SAT', num_workers=1)
    if result:
        total_value, solution, total_weight, wall_time, conflicts = result
        print(f"Total value: {total_value}")
        print(f"Total weight: {total_weight}")

    # 멀티스레드 테스트
    for workers in [1, 4, 8, 16, 24, 32, 64, 128]:
        print(f"\n=== Testing with {workers} workers ===")
        result = solve_fractional_knapsack_or(values, weights, capacity, 1, solver_name='CP-SAT', num_workers=workers)
        if result:
            _, _, _, wall_time, conflicts = result
            print(f"{workers} workers → WallTime: {wall_time:.4f} sec, Conflicts: {conflicts}")
