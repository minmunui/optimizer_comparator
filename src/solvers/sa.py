import random
import math
from typing import Callable, List

import matplotlib
from scipy.optimize import dual_annealing

from src.solvers.solver import Solver

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# If needed, you can inherit from a base Solver class, e.g., from src.solvers.solver import Solver

class SimulatedAnnealingSolver(Solver):  # Optionally, inherit from Solver if required.
    def __init__(self,
                 solution_type: List[int],
                 fitness_function: Callable[[List[int]], float],
                 initial_temp: float = 1000,
                 cooling_rate: float = 0.995,
                 min_temp: float = 1e-3,
                 max_iter: int = 10000):
        """
        :param solution_type: List indicating the range for each variable (for Knapsack, usually 0 or 1)
        :param fitness_function: Function to evaluate the fitness of a solution. Higher values indicate better solutions.
        :param initial_temp: Starting temperature for the simulated annealing algorithm.
        :param cooling_rate: Rate at which the temperature is decreased (between 0 and 1).
        :param min_temp: Minimum temperature threshold to stop the algorithm.
        :param max_iter: Maximum number of iterations.
        """
        self.solution_type = solution_type
        self.fitness_function = fitness_function
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter
        self.solution_length = len(solution_type)
        self.history = []  # Records the progress of the algorithm.
        self.best_solution = None

    def random_solution(self) -> List[int]:
        """
        Generate a random initial solution.
        Each variable is chosen randomly within the range specified by solution_type.
        For the Knapsack problem, each element is typically 0 or 1.
        """
        return [random.randint(0, self.solution_type[i]) for i in range(self.solution_length)]

    def get_neighbor(self, solution: List[int]) -> List[int]:
        """
        Generate a neighboring solution by flipping the selection of a random item.
        """
        neighbor = solution.copy()
        i = random.randint(0, self.solution_length - 1)
        neighbor[i] = 1 - neighbor[i]  # Flip between 0 and 1.
        return neighbor

    def solve(self) -> List[int]:
        """
        Execute the Simulated Annealing algorithm to find an optimal or near-optimal solution.
        The algorithm's progress is recorded in the history.
        """
        current = self.random_solution()
        current_fitness = self.fitness_function(current)
        best = current
        best_fitness = current_fitness
        T = self.initial_temp
        iter_count = 0

        while T > self.min_temp and iter_count < self.max_iter:
            neighbor = self.get_neighbor(current)
            neighbor_fitness = self.fitness_function(neighbor)
            delta = neighbor_fitness - current_fitness

            # Always accept better solutions; for worse solutions, accept with a probability.
            if delta > 0 or random.random() < math.exp(delta / T):
                current = neighbor
                current_fitness = neighbor_fitness
                if current_fitness > best_fitness:
                    best = current
                    best_fitness = current_fitness

            # Record the progress at each iteration.
            self.history.append({
                "iter": iter_count,
                "temperature": T,
                "current_fitness": current_fitness,
                "best_fitness": best_fitness
            })

            T *= self.cooling_rate
            iter_count += 1

        self.best_solution = best
        return best

    def plot_history(self, save_path: str = None):
        """
        Plot the history of the algorithm's progress based on the recorded best fitness over iterations.
        :param save_path: If provided, the plot is saved to the given file path.
        """
        iterations = [record['iter'] for record in self.history]
        best_fitness = [record['best_fitness'] for record in self.history]
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_fitness, label='Best Fitness')
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Simulated Annealing Progress")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class DualAnnealingSolver(Solver):

    def __init__(self,
                 solution_type: list[int],
                 fitness_function: Callable[[list[int]], float],
                 initial_temp: float = 10460,
                 maxiter: int = 10000,
                 max_fun: int = None,
                 ):
        self.max_fun = max_fun if max_fun is not None else maxiter * 10
        self.solution_type = solution_type
        self.fitness_function = fitness_function
        self.solution_length = len(solution_type)
        self.maxiter = maxiter
        self.history = []
        self.initial_temp = initial_temp

    def random_solution(self) -> list[int]:
        return [random.randint(0, self.solution_type[i]) for i in range(self.solution_length)]

    def solve(self) -> list[int]:
        bounds = [(0, action) for action in self.solution_type]
        result = dual_annealing(self.fitness_function, bounds=bounds, maxiter=self.maxiter, initial_temp=self.initial_temp)
        return result.x
