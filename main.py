import argparse
import random
import time
from enum import Enum

from src.solvers.or_knapsack import solve_fractional_knapsack_or
from src.problems.knapsack import generate_knapsack_problem, eval_knapsack, eval_knapsack_rag
from src.solvers.ga import MyGaSolver, ParentSelectionMethod
from src.solvers.sa import DualAnnealingSolver
from src.utils import logger
from src.solvers.dp import solve_dp_fraction


class SolverType(Enum):
    DP = "dp"
    GA = "ga"
    GA_RAG = "ga-rag"
    GA_LIB = "ga-lib"
    DA = "da"
    OR_TOOLS = "or-tools"

    @classmethod
    def from_string(cls, value):
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid solver: {value}. Must be one of: {', '.join(valid_values)}")


def log_solution(values, weights, choices, fraction=1):
    str_weights = ''.join([f'{w:<6}' for w in weights])
    str_values = ''.join([f'{v:<6}' for v in values])
    str_choices = ''.join([f'{int(c):<6}' for c in choices])
    str_gotten_values = ''.join([f'{values[i] * choices[i] / fraction:<6}' for i in range(len(values))])
    logger.info(f"{'Values':<10}: {str_values}")
    logger.info(f"{'Weights':<10}: {str_weights}")
    logger.info(f"{'Solution':<10}: {str_choices}")
    logger.info(f"{'Got values':<10}: {str_gotten_values}")
    logger.info(f"Total value: {sum([values[i] * choices[i] / fraction for i in range(len(values))])}")
    logger.info(f"Total weight: {sum([weights[i] * choices[i] / fraction for i in range(len(weights))])}")


def solve_with_timer(solver_func, *args, **kwargs):
    """Execute solver function with timer and return results"""
    time_start = time.time()
    result = solver_func(*args, **kwargs)
    time_end = time.time()
    logger.info(f'Elapsed time: {time_end - time_start:.4f} seconds')
    return result


def solve_or_tools(values, weights, capacity, fraction):
    logger.info("OR-Tools Solution")
    total_value, solution, packed_weights = solve_with_timer(
        solve_fractional_knapsack_or,
        values, weights, capacity, fraction
    )
    log_solution(values, weights, solution, fraction)


def solve_dynamic_programming(values, weights, capacity, fraction):
    logger.info("Dynamic Programming Solution")
    total_value, actual_weight, choices = solve_with_timer(
        solve_dp_fraction,
        weights, values, capacity, fraction
    )
    log_solution(values, weights, choices, fraction)


def solve_genetic_algorithm(values, weights, capacity, fraction, use_rag=False,
                            population_size=1000,
                            num_parents=250,
                            immigration_size=200,
                            parent_select_method="Tournament",
                            crossover_rate: float = 0.8,
                            mutation_rate: float = 0.1,
                            ):
    logger.info("Genetic Algorithm Solution")
    if use_rag:
        fitness_function = lambda x: eval_knapsack_rag(x, values=values, weights=weights,
                                                       capacity=capacity, division=fraction)
    else:
        fitness_function = lambda x: eval_knapsack(x, values=values, weights=weights,
                                                   capacity=capacity, division=fraction)
    input_type = [fraction for _ in range(len(weights))]

    # Convert string to ParentSelectionMethod enum
    try:
        selection_method = ParentSelectionMethod[parent_select_method]
    except KeyError:
        selection_method = ParentSelectionMethod.Tournament

    ga_solver = MyGaSolver(
        population_size=population_size,
        solution_type=input_type,
        fitness_function=fitness_function,
        immigration_size=immigration_size,
        num_parents=num_parents,
        parent_select_method=selection_method,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    best_solution = solve_with_timer(ga_solver.solve, max_generations=200)
    ga_solver.plot_history()
    log_solution(values, weights, best_solution, fraction)


def solve_genetic_algorithm_lib(values, weights, capacity, fraction,
                                num_generations=200, num_parents_mating=250, sol_per_pop=1000,
                                mutation_percent_genes=10, keep_parents=2):
    logger.info("PyGAD Solution")
    import pygad

    fitness_function = lambda _, x, y: eval_knapsack(x, values=values, weights=weights,
                                                     capacity=capacity, fraction=fraction)
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=len(weights),
        init_range_low=0,
        init_range_high=1,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=mutation_percent_genes,
        keep_parents=keep_parents,
        gene_space=[n for n in range(fraction + 1)]
    )

    solve_with_timer(ga_instance.run)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    log_solution(values, weights, solution, fraction)


def solve_dual_annealing(values, weights, capacity, fraction,
                         initial_temp=10460, maxiter=10000):
    logger.info("Dual Annealing Solution")
    fitness_function = lambda x: -1 * eval_knapsack(x, values=values, weights=weights,
                                                    capacity=capacity, division=fraction)
    sa_solver = DualAnnealingSolver(
        solution_type=[fraction for _ in range(len(weights))],
        fitness_function=fitness_function,
        initial_temp=initial_temp,
        maxiter=maxiter
    )

    best_solution = solve_with_timer(sa_solver.solve)
    best_solution = [round(s) for s in best_solution]
    log_solution(values, weights, best_solution, fraction)


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration
    parser.add_argument("--num_items", type=int, default=50, help="Number of items")
    parser.add_argument("--weight_range", type=int, nargs=2, default=[1, 50], help="Range of item weights (min, max)")
    parser.add_argument("--value_range", type=int, nargs=2, default=[10, 500], help="Range of item values (min, max)")
    parser.add_argument("--capacity_ratio", type=float, default=0.3,
                        help="Knapsack capacity as a ratio of total item weight")
    parser.add_argument("--fraction", type=int, default=1,
                        help="Fractional knapsack ratio. if 1, normal knapsack problem")
    parser.add_argument("--solver", type=str, default="dp",
                        help=f"Solver to use: {', '.join([e.value for e in SolverType])}")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")

    # GA hyperparameters
    parser.add_argument("--ga_population_size", type=int, default=1000, help="Population size for GA solver")
    parser.add_argument("--ga_num_parents", type=int, default=250, help="Number of parents for GA solver")
    parser.add_argument("--ga_immigration_size", type=int, default=200, help="Immigration size for GA solver")
    parser.add_argument("--ga_parent_select_method", type=str, default="Tournament",
                        help="Parent selection method for GA solver")

    # GA_LIB hyperparameters
    parser.add_argument("--ga_lib_num_generations", type=int, default=200,
                        help="Number of generations for GA LIB solver")
    parser.add_argument("--ga_lib_num_parents_mating", type=int, default=250,
                        help="Number of parents mating for GA LIB solver")
    parser.add_argument("--ga_lib_sol_per_pop", type=int, default=1000,
                        help="Solution per population for GA LIB solver")
    parser.add_argument("--ga_lib_mutation_percent_genes", type=int, default=10,
                        help="Mutation percent genes for GA LIB solver")
    parser.add_argument("--ga_lib_keep_parents", type=int, default=2,
                        help="Number of parents to keep for GA LIB solver")

    # Dual Annealing hyperparameters
    parser.add_argument("--da_initial_temp", type=float, default=10460,
                        help="Initial temperature for Dual Annealing solver")
    parser.add_argument("--da_maxiter", type=int, default=10000, help="Maximum iterations for Dual Annealing solver")

    args = parser.parse_args()

    if args.log_file:
        logger.get_logger().start_file_logging(args.log_file)

    random.seed(args.random_seed)

    weights, values, capacity = generate_knapsack_problem(
        num_items=args.num_items,
        weight_range=args.weight_range,
        value_range=args.value_range,
        capacity_ratio=args.capacity_ratio
    )
    logger.get_logger().set_timestamp(False)

    logger.info(f"Fraction: {args.fraction}")
    logger.info(f"Capacity: {capacity}")
    logger.info(f"Weights: {weights}")
    logger.info(f"Values: {values}")

    # Convert string to enum and validate input
    solver_type = SolverType.from_string(args.solver)

    # Dispatch to appropriate solver with hyperparameters
    solvers = {
        SolverType.OR_TOOLS: lambda: solve_or_tools(values, weights, capacity, args.fraction),
        SolverType.DP: lambda: solve_dynamic_programming(values, weights, capacity, args.fraction),
        SolverType.GA: lambda: solve_genetic_algorithm(
            values, weights, capacity, args.fraction,
            use_rag=False,
            population_size=args.ga_population_size,
            num_parents=args.ga_num_parents,
            immigration_size=args.ga_immigration_size,
            parent_select_method=args.ga_parent_select_method,
            crossover_rate=args.ga_crossover_rate,
            mutation_rate=args.ga_mutation_rate
        ),
        SolverType.GA_RAG: lambda: solve_genetic_algorithm(
            values, weights, capacity, args.fraction,
            use_rag=True,
            population_size=args.ga_population_size,
            num_parents=args.ga_num_parents,
            immigration_size=args.ga_immigration_size,
            parent_select_method=args.ga_parent_select_method,
            crossover_rate=args.ga_crossover_rate,
            mutation_rate=args.ga_mutation_rate
        ),
        SolverType.GA_LIB: lambda: solve_genetic_algorithm_lib(
            values, weights, capacity, args.fraction,
            num_generations=args.ga_lib_num_generations,
            num_parents_mating=args.ga_lib_num_parents_mating,
            sol_per_pop=args.ga_lib_sol_per_pop,
            mutation_percent_genes=args.ga_lib_mutation_percent_genes,
            keep_parents=args.ga_lib_keep_parents
        ),
        SolverType.DA: lambda: solve_dual_annealing(
            values, weights, capacity, args.fraction,
            initial_temp=args.da_initial_temp,
            maxiter=args.da_maxiter
        ),
    }

    # Execute selected solver
    solvers[solver_type]()


if __name__ == "__main__":
    main()
