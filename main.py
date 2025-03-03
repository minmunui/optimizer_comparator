from src.problems.knapsack import generate_knapsack_problem, evalKnapsack
from src.solvers.genetic_alorithm import GeneticSolver

weights, values, capacity = generate_knapsack_problem()

input_type = [1 for _ in range(len(weights))]

ga_solver = GeneticSolver(
    population_size=1000,
    input_type=input_type,
    fitness_function=lambda x: evalKnapsack(x, values=values, weights=weights, capacity=capacity)
)

ga_solver.solve(max_generations=200)
ga_solver.plot_history()
