from src.problems.knapsack import evalKnapsack, generate_knapsack_problem
import time

from src.solvers.ga import CrossoverMethod, ParentSelectionMethod, MyGaSolver

candidate = {
    "cross_methods": [CrossoverMethod.Random],
    "parent_select_methods": [ParentSelectionMethod.Tournament],
    "population_size": [1000],
    "parents_size": [50/100.0],
    "immigration_size": [10/100.0],
    "num_generations": [150]
}

weights, values, capacity = generate_knapsack_problem()

print(f"Knapsack Problem Information:")
print(f"weights: {weights}")
print(f"Knapsack capacity: {capacity}")
print(f"values: {values}")


fitness_function = lambda x: evalKnapsack(x, values=values, weights=weights, capacity=capacity)
print(f"{'Cross':<15}{'Parent':<15}{'Pop':<10}{'num_gen':<10}{'Parent':<10}{'Imm':<10}{'Fitness':<10}{'Time':<10}")
for cross_method in candidate["cross_methods"]:
    for parent_method in candidate["parent_select_methods"]:
        for pop_size in candidate["population_size"]:
            for parent_size in candidate["parents_size"]:
                for imm_size in candidate["immigration_size"]:
                    for num_gen in candidate["num_generations"]:
                        solver = MyGaSolver(
                            population_size=pop_size,
                            num_parents=int(pop_size * parent_size),
                            num_elitism=2,
                            crossover_method=cross_method,
                            parent_select_method=parent_method,
                            immigration_size=int(imm_size * pop_size),
                            fitness_function=fitness_function,
                            solution_type=[1 for _ in range(len(weights))],
                        )
                        start_time = time.time()
                        best_solution = solver.solve(max_generations=num_gen)
                        end_time = time.time()
                        best_fitness = max(solver.calculate_all_fitness())
                        elapsed_time = end_time - start_time
                        file_name = f"GA_{best_fitness:.2f}_{elapsed_time:.2f}_{cross_method}_{parent_method}_{pop_size}_{parent_size}_{imm_size}.png"
                        solver.plot_history(save_path=file_name)
                        print(f"{cross_method:<15}{parent_method:<15}{pop_size:<10}{num_gen:<10}{parent_size:<10.2f}{imm_size:<10.2f}{best_fitness:<10}{elapsed_time:<10.2f}")