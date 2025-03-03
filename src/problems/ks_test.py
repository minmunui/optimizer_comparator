import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from src.problems.knapsack import generate_knapsack_problem, evalKnapsack

# Generate the problem
weights, values, capacity = generate_knapsack_problem()

# Print problem information
print(f"Knapsack Problem Information:")
print(f"Number of items: 80")
print(f"Knapsack capacity: {capacity}")
print("\nItem details (first 10 items):")
print("Index\tWeight\tValue\tValue/Weight Ratio")
for i in range(10):  # Only print the first 10 items
    ratio = values[i] / weights[i]
    print(f"{i}\t{weights[i]}\t{values[i]}\t{ratio:.2f}")

# Solve the problem using Genetic Algorithm
# Using DEAP library

# Define fitness class (maximization problem)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Set up genetic algorithm toolbox
toolbox = base.Toolbox()

# Individual creator: decide whether to select(1) or not select(0) each item
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(weights))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)




# Register genetic operators
toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutation probability 5%
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection


# Run the genetic algorithm
def run_ga(population_size=100, generations=100):
    # Create initial population
    population = toolbox.population(n=population_size)

    # Set up statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run genetic algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.7,  # Crossover probability 70%
        mutpb=0.2,  # Mutation probability 20%
        ngen=generations,
        stats=stats,
        halloffame=tools.HallOfFame(1),  # Preserve 1 best individual
        verbose=True
    )

    return population, logbook


# Run genetic algorithm
population, logbook = run_ga()

# Check best individual
best_ind = tools.selBest(population, 1)[0]
total_value = 0
total_weight = 0
selected_items = []

for i, item in enumerate(best_ind):
    if item == 1:
        total_value += values[i]
        total_weight += weights[i]
        selected_items.append(i)

print("\nOptimal Solution:")
print(f"Total value: {total_value}")
print(f"Total weight: {total_weight}/{capacity}")
print(f"Number of selected items: {len(selected_items)}/{len(weights)}")
print(f"Selected item indices (first 10): {selected_items[:10]}...")

# Plot performance graph
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
plt.figure(figsize=(10, 6))
plt.plot(gen, avg, label="Average")
plt.plot(gen, min_, label="Minimum")
plt.plot(gen, max_, label="Maximum")
plt.title("Fitness Evolution Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness (Value)")
plt.legend()
plt.grid(True)
plt.show()

# Analyze population
fitness_values = [ind.fitness.values[0] for ind in population]
plt.figure(figsize=(10, 6))
plt.hist(fitness_values, bins=30)
plt.title("Fitness Distribution in Final Population")
plt.xlabel("Fitness (Value)")
plt.ylabel("Number of Individuals")
plt.grid(True)
plt.show()

# Parameter sensitivity analysis
# def analyze_parameters():
#     results = []
#
#     # Vary population size
#     population_sizes = [50, 100, 200]
#     for pop_size in population_sizes:
#         _, logbook = run_ga(population_size=pop_size, generations=50)
#         best_fitness = max(logbook.select("max"))
#         results.append((f"Population Size={pop_size}", best_fitness))
#
#     # Vary crossover rate
#     crossover_rates = [0.5, 0.7, 0.9]
#     for cxpb in crossover_rates:
#         # Run custom algorithm
#         population = toolbox.population(n=100)
#         _, logbook = algorithms.eaSimple(
#             population, toolbox, cxpb=cxpb, mutpb=0.2, ngen=50,
#             stats=stats, halloffame=tools.HallOfFame(1), verbose=False
#         )
#         best_fitness = max(logbook.select("max"))
#         results.append((f"Crossover Rate={cxpb}", best_fitness))
#
#     # Print results
#     for param, fitness in results:
#         print(f"{param}: Best Fitness = {fitness}")

# Run sensitivity analysis (commented out as it may take a long time)
# analyze_parameters()
