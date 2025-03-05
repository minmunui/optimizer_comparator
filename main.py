import argparse
import time

from src.solvers.or_knapsack import solve_knapsack_or, solve_fractional_knapsack_or
from src.problems.knapsack import generate_knapsack_problem, evalKnapsack, eval_fractional_knapsack
from src.solvers.ga import MyGaSolver, ParentSelectionMethod
from src.utils import logger
from src.solvers.dp import solve_dp, solve_dp_fraction


def log_solution(values, weights, choices, fraction=1):
    str_weights = ''.join([f'{w:<6}' for w in weights])
    str_values = ''.join([f'{v:<6}' for v in values])
    str_choices = ''.join([f'{int(c):<6}' for c in choices])
    str_gotten_values = ''.join([f'{values[i] * choices[i] / fraction:<6}' for i in range(len(values))])
    logger.info(f"{'Values':<10}: {str_values}")
    logger.info(f"{'Weights':<10}: {str_weights}")
    logger.info(f"{'Solution':<10}: {str_choices}")
    logger.info(f"{'Got values':<10}: {str_gotten_values}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=50, help="Number of items")
    parser.add_argument("--weight_range", type=int, nargs=2, default=[1, 50], help="Range of item weights (min, max)")
    parser.add_argument("--value_range", type=int, nargs=2, default=[10, 500], help="Range of item values (min, max)")
    parser.add_argument("--capacity_ratio", type=float, default=0.3,
                        help="Knapsack capacity as a ratio of total item weight")
    parser.add_argument("--fraction", type=int, default=1,
                        help="Fractional knapsack ratio. if 1, normal knapsack problem")
    parser.add_argument("--solver", type=str, default="dp", help="Solver to use (dp, ga, or-tools)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    weights, values, capacity = generate_knapsack_problem(num_items=args.num_items,
                                                          weight_range=args.weight_range,
                                                          value_range=args.value_range,
                                                          capacity_ratio=args.capacity_ratio)
    logger.get_logger().set_timestamp(False)

    if args.fraction == 1:
        logger.info(f"Fractional Knapsack Problem")

        if args.solver == "dp":
            logger.info(f"Dynamic Programming Solution")
            time_start = time.time()
            total_value, actual_weight, choices = solve_dp(weights, values, capacity)
            logger.info(solve_dp(weights, values, capacity))
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {total_value}")
            log_solution(values, weights, choices)

        elif args.solver == "ga":
            logger.info(f"Genetic Algorithm Solution")
            fitness_function = lambda x: evalKnapsack(x, values=values, weights=weights, capacity=capacity)
            input_type = [1 for _ in range(len(weights))]
            time_start = time.time()
            ga_solver = MyGaSolver(
                population_size=1000,
                solution_type=input_type,
                fitness_function=fitness_function,
                immigration_size=200,
                num_parents=250,
                parent_select_method=ParentSelectionMethod.Tournament
            )
            best_solution = ga_solver.solve(max_generations=200)
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {ga_solver.calculate_fitness(best_solution)}")
            log_solution(values, weights, best_solution)
            ga_solver.plot_history()

        elif args.solver == "or-tools":
            logger.info(f"OR-Tools Solution")
            time_start = time.time()
            total_value, solution, packed_weights = solve_knapsack_or(values, weights, capacity)
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {total_value}")
            log_solution(values, weights, solution)

        else:
            raise ValueError(f"Solver should be one of dp, ga, or or-tools: {args.solver}")

    elif args.fraction > 1:
        logger.info(f"Fractional Knapsack Problem")
        logger.info(f"Fraction: {args.fraction}")
        logger.info(f"Capacity: {capacity}")
        logger.info(f"Weights: {weights}")
        logger.info(f"Values: {values}")

        if args.solver == "or-tools":
            logger.info(f"OR-Tools Solution")
            time_start = time.time()
            total_value, solution, packed_weights = solve_fractional_knapsack_or(values, weights, capacity,
                                                                                 args.fraction)
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {total_value}")
            log_solution(values, weights, solution, args.fraction)

        elif args.solver == "dp":
            logger.info(f"Dynamic Programming Solution")
            time_start = time.time()
            total_value, actual_weight, choices = solve_dp_fraction(weights, values, capacity, args.fraction)
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {total_value}")
            log_solution(values, weights, choices, args.fraction)

        elif args.solver == "ga":
            logger.info(f"Genetic Algorithm Solution")
            fitness_function = lambda x: eval_fractional_knapsack(x, values=values, weights=weights, capacity=capacity,
                                                                  division=args.fraction)
            input_type = [1 for _ in range(len(weights))]
            time_start = time.time()
            ga_solver = MyGaSolver(
                population_size=1000,
                solution_type=input_type,
                fitness_function=fitness_function,
                immigration_size=200,
                num_parents=250,
                parent_select_method=ParentSelectionMethod.Tournament
            )
            best_solution = ga_solver.solve(max_generations=200)
            time_end = time.time()
            logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
            logger.info(f"Total value: {ga_solver.calculate_fitness(best_solution)}")
            ga_solver.plot_history()
            log_solution(values, weights, best_solution, args.fraction)
        else:
            raise ValueError(f"Solver should be one of dp, ga, or or-tools: {args.solver}")

    else:
        raise ValueError(f"Fraction should be greater than 0: {args.fraction}")


if __name__ == "__main__":
    main()
