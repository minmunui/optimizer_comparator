def solve_dp(weights: list[int], values: list[int], capacity: int):
    n = len(weights)  # Number of items
    # dp[i][w]: Maximum value achievable using the first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtracking to reconstruct the chosen items
    choices = [0] * n
    w = capacity
    for i in range(n, 0, -1):
        # If the current value is different from the value without the i-th item,
        # then the i-th item was selected.
        if dp[i][w] != dp[i - 1][w]:
            choices[i - 1] = 1
            w -= weights[i - 1]
        else:
            choices[i - 1] = 0

    actual_profit = dp[n][capacity]
    actual_weight = sum(weights[i] * choices[i] for i in range(n))

    return actual_profit, actual_weight, choices


def solve_dp_fraction(weights: list[int], values: list[int], capacity: int, fraction: int):
    """
    각 아이템에 대해 0, 1/n, ..., 1 선택지를 고려하는 knapsack 문제를 동적 계획법으로 풉니다.
    weights: 각 아이템의 무게 (정수)
    profits: 각 아이템의 이익 (정수)
    capacity: 배낭의 최대 용량 (정수)

    반환값:
      actual_profit: 실제 최대 이익 (소수점 포함, 원 단위)
      actual_weight: 실제 사용한 무게 (원 단위)
      choices: 각 아이템에 대해 선택한 옵션 (0~n, 실제 담은 비율은 choices[i]/n)
    """
    # 분수 문제를 정수로 다루기 위해 배낭 용량을 3배 확장
    cap_scaled = capacity * fraction
    n = len(weights)

    # dp[w]: 현재까지 사용한 용량이 w일 때 얻을 수 있는 최대 "스케일된" 이익
    dp = [-1] * (cap_scaled + 1)
    dp[0] = 0

    # reconstruction을 위한 배열들
    # decision[i][w]: i번째 아이템 선택 후, 현재 용량 w에 대해 선택한 옵션 s (0,1,2,3)
    decision = [[-1] * (cap_scaled + 1) for _ in range(n)]
    # prev[i][w]: i번째 아이템 선택 전의 용량
    prev = [[-1] * (cap_scaled + 1) for _ in range(n)]

    # 각 아이템에 대해 DP 갱신
    for i in range(n):
        new_dp = [-1] * (cap_scaled + 1)
        for w in range(cap_scaled + 1):
            if dp[w] >= 0:
                # 각 선택지 s: 0(안 담음), 1(1/n 담음), ..., n(전부 담음)
                for s in range(fraction + 1):
                    new_w = w + s * weights[i]
                    if new_w <= cap_scaled:
                        candidate = dp[w] + s * values[i]
                        if candidate > new_dp[new_w]:
                            new_dp[new_w] = candidate
                            decision[i][new_w] = s
                            prev[i][new_w] = w
        dp = new_dp

    # 가능한 모든 용량 사용 중 최대 이익을 선택
    best_profit = max(dp)
    best_w = dp.index(best_profit)

    # 각 아이템에 대해 선택한 옵션을 역추적
    choices = [0] * n
    w = best_w
    for i in range(n - 1, -1, -1):
        s = decision[i][w]
        choices[i] = s
        w = prev[i][w] if i > 0 else 0

    # dp에 저장된 이익과 무게는 3배 스케일된 값이므로 실제 값은 1/3을 곱해 복원
    actual_profit = best_profit / fraction
    actual_weight = sum(choices[i] * weights[i] for i in range(n)) / fraction

    return actual_profit, actual_weight, choices
