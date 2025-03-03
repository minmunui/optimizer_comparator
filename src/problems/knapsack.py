import random


# Problem generation function
def generate_knapsack_problem(num_items: int = 80,
                              weight_range: tuple[int, int] = (1, 50),
                              value_range: tuple[int, int] = (10, 500),
                              capacity_ratio: float = 0.3) \
        -> tuple[list[int], list[int], int]:
    """
    Generate a knapsack problem dataset.

    Args:
        num_items: Number of items
        weight_range: Range of item weights (min, max)
        value_range: Range of item values (min, max)
        capacity_ratio: Knapsack capacity as a ratio of total item weight

    Returns:
        Tuple of (weights, values, capacity)
    """
    # Set random seed for reproducible results
    random.seed(42)

    # Generate random weights and values for items
    weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(num_items)]
    values = [random.randint(value_range[0], value_range[1]) for _ in range(num_items)]

    # Set knapsack capacity (as a ratio of total weight)
    total_weight = sum(weights)
    capacity = int(total_weight * capacity_ratio)

    return weights, values, capacity


# Define fitness function
def evalKnapsack(individual, values, weights, capacity):
    total_value = 0
    total_weight = 0
    for i, item in enumerate(individual):
        if item == 1:  # If the item is selected
            total_value += values[i]
            total_weight += weights[i]

    # Apply penalty for exceeding capacity
    if total_weight > capacity:
        return 0,  # comma indicates tuple return
    return total_value,

