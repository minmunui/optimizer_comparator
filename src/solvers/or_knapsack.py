import time

from ortools.linear_solver import pywraplp

from src.problems.knapsack import generate_knapsack_problem


def solve_knapsack_or(values: list[int], weights: list[int], capacity: int):
    # 솔버 생성
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # 변수 생성 (각 아이템을 선택할지 말지를 나타내는 이진 변수)
    x = {}
    for i in range(len(values)):
        x[i] = solver.IntVar(0, 1, f'x_{i}')

    # 제약 조건 추가 (총 무게는 용량을 초과할 수 없음)
    solver.Add(sum(weights[i] * x[i] for i in range(len(weights))) <= capacity)

    # 목적 함수 설정 (총 가치 최대화)
    objective = solver.Objective()
    for i in range(len(values)):
        objective.SetCoefficient(x[i], values[i])
    objective.SetMaximization()

    # 문제 해결
    status = solver.Solve()

    # 결과 추출
    if status == pywraplp.Solver.OPTIMAL:
        packed_items = []
        packed_weights = 0
        total_value = 0
        solution = [x[i].solution_value() for i in range(len(values))]
        for i in range(len(values)):
            packed_weights += weights[i]
            total_value += values[i] * solution[i]
        return total_value, solution, packed_weights
    return None, None, None


def solve_fractional_knapsack_or(weights: list[int], values: list[int], capacity: int, fraction: int):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP 솔버를 생성할 수 없습니다.")
        return

    n = len(weights)
    # 각 물건에 대해 0, 1, 2, 3 중 하나의 값을 갖는 정수 변수 x[i] 생성
    # 여기서 x[i] / 3 이 실제 담는 비율을 나타냅니다.
    x = [solver.IntVar(0, fraction, f'x{i}') for i in range(n)]

    # constraint: sum( weights[i] * x[i] ) <= capacity * fraction
    solver.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity * fraction)

    # objective: maximize sum( values[i] * x[i] )
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], values[i])
    objective.SetMaximization()

    # 문제 풀기
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # 실제 총 수익과 무게는 각 변수 값에 1/3을 곱한 결과임
        total_value = sum(values[i] * x[i].solution_value() for i in range(n)) / fraction
        total_weight = sum(weights[i] * x[i].solution_value() for i in range(n)) / fraction
        for i in range(n):
            x[i] = x[i].solution_value()
        return total_value, x, total_weight
    else:
        print('최적 해를 찾지 못했습니다.')
