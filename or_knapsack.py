from ortools.linear_solver import pywraplp


def solve_knapsack(values, weights, capacity):
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
        for i in range(len(values)):
            if x[i].solution_value() > 0.5:  # 반올림 오차 방지
                packed_items.append(i)
                packed_weights += weights[i]
                total_value += values[i]
        return total_value, packed_items, packed_weights
    return None, None, None


# 예제 사용
values = [60, 100, 120]  # 각 아이템의 가치
weights = [10, 20, 30]  # 각 아이템의 무게
capacity = 50  # 배낭의 최대 용량

total_value, selected_items, total_weight = solve_knapsack(values, weights, capacity)
if total_value is not None:
    print(f'총 가치: {total_value}')
    print(f'선택된 아이템: {selected_items}')
    print(f'총 무게: {total_weight}')
else:
    print('해결책을 찾을 수 없습니다.')