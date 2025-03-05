import random
from enum import Enum
from typing import Callable

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from src.solvers.solver import Solver
from src.utils import logger


class CrossoverMethod(Enum):
    Sequential = 1,
    Iterative = 2,
    Random = 3

    def __str__(self):
        return self.name


class ParentSelectionMethod(Enum):
    RouletteWheel = 1,
    Tournament = 2

    def __str__(self):
        return self.name


class MyGaSolver(Solver):
    def __init__(self,
                 solution_type: list[int],
                 fitness_function: Callable[[list[int]], float],
                 population_size: int = 100,
                 num_parents: int = None,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = -1,
                 num_elitism: int = 2,
                 immigration_size: int = None,
                 parent_select_method: ParentSelectionMethod = ParentSelectionMethod.Tournament,
                 crossover_method: CrossoverMethod = CrossoverMethod.Sequential,
                 ):
        """


        :param solution_type: array of input ranges
        :param fitness_function: fitness function to evaluate the solution. as higher the value, the better the solution
        :param population_size: population size of the genetic algorithm (normally 50~200)
        :param num_parents: number of parents to be selected for crossover (normally 2)
        :param crossover_rate: probability of crossover between parents (normally 0.7~0.9)
        :param mutation_rate: probability of mutation of a solution (normally 0.001~0.05)
        :param num_elitism: number of the best solutions to be passed to the next generation
        :param immigration_size: number of new solutions to be added to the population in each generation
        """

        self.history = []
        self.immigration_size = immigration_size if immigration_size >= 0 else population_size // 10
        self.solution_length = len(solution_type)
        self.solution_type = solution_type
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.num_parents = num_parents if num_parents is not None else population_size // 4
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate if mutation_rate >= 0 else 1 / self.solution_length
        self.num_elitism = num_elitism

        self.population = self.generate_population()

        self.parent_select_method = parent_select_method
        self.crossover_method = crossover_method

    def init(self):
        """
        Initialize the population
        :return:
        """
        self.population = self.generate_population()

    def set_population(self, population: list[list[int]]):
        """
        Set the population of the genetic algorithm
        :param population:
        :return:
        """
        self.population = population

    def set_parameters(self, **kwargs):
        """
        Set the parameters of the genetic algorithm
        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate_population(self, num_population: int = 0) -> list[list[int]]:
        """
        Generate a population of random solutions
        :return:
        """
        num_population = self.population_size if num_population == 0 else num_population
        return [[random.randint(0, self.solution_type[i]) for i in self.solution_type]
                for _ in range(num_population)]

    def calculate_fitness(self, solution: list[int]) -> float:
        """
        Calculate the fitness of a solution
        :param solution:
        :return:
        """
        return self.fitness_function(solution)[0]

    def calculate_all_fitness(self) -> list[float]:
        """
        Calculate the fitness of all solutions in the population
        :return:
        """
        return [self.calculate_fitness(solution) for solution in self.population]

    def select_parents_roulette_wheel(self, k: int, fitness: list[int] | None = None) -> list[list[int]]:
        """
        Select K parents from the population based on their fitness using roulette wheel selection
        :param k: number of parents to be selected
        :param fitness: list of fitness values of the population, if None, calculate the fitness in the function
        :return:
        """
        fitness = self.calculate_all_fitness() if fitness is None else fitness
        total_fitness = sum(fitness)
        if total_fitness == 0:
            return random.choices(self.population, k=k)
        probabilities = [f / total_fitness for f in fitness]

        parents = random.choices(self.population, probabilities, k=k)
        return parents

    # 다양한 부모선택 알고리즘 변경하기
    def select_parents_tournament(self, n_selection: int, fitness: list[float] | None = None,
                                  tournament_size: int = 5) -> \
            list[list[int]]:
        """
        Select two parents from the population based on their fitness using tournament selection
        :param fitness: list of fitness values of the population, if None, calculate the fitness in the function
        :param n_selection: number of solutions to be selected as parents
        :param tournament_size: number of solutions to be selected for the tournament
        :return:
        """
        parents = []
        fitness = self.calculate_all_fitness() if fitness is None else fitness
        for _ in range(n_selection):
            tournament = random.sample(range(len(self.population)), tournament_size)
            best_index = max(tournament, key=lambda x: fitness[x])
            parents.append(self.population[best_index])
        return parents

    def select_elites(self, n_elites: int, fitness: list[float] | None = None) -> list[list[int]]:
        """
        Select the best solutions from the population
        :param fitness: list of fitness values of the population, if None, calculate the fitness in the function
        :param n_elites: number of elite solutions to be selected
        :return:
        """
        fitness = self.calculate_all_fitness() if fitness is None else fitness
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:n_elites]
        return [self.population[i] for i in elite_indices]

    def select_parents(self,
                       n_selection: int,
                       method: ParentSelectionMethod,
                       fitness: list[float] | None = None,
                       **kwargs) -> list[list[int]]:
        """
        Select parents based on the selection method
        :param fitness: list of fitness values of the population, if None, calculate the fitness in the function
        :param n_selection: number of solutions to be selected as parents
        :param method: selection method
        :param kwargs: additional arguments for the selection method
        :return:
        """
        if method == ParentSelectionMethod.RouletteWheel:
            return self.select_parents_roulette_wheel(n_selection, fitness)
        elif method == ParentSelectionMethod.Tournament:
            return self.select_parents_tournament(n_selection, fitness, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Crossover two parents to create 2 children, order is not important
        :param parent1: first parent
        :param parent2: second parent
        :return:
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.solution_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def crossover_population(self,
                             parents: list[list[int]],
                             crossover_method: CrossoverMethod,
                             num_children: int = None) -> list[list[int]]:

        """
        Crossover the parents to create children
        :param parents: list of parents to generate children
        :param crossover_method: method of crossover
        :param num_children: number of children to be generated
        :return:
        """
        children = []
        random.shuffle(parents)
        if crossover_method == CrossoverMethod.Sequential or crossover_method == CrossoverMethod.Iterative:
            index = 0
            temp = []
            while num_children > len(children):
                temp.append(parents[index])
                index = index + 1 % len(parents)
                if len(temp) >= 2:
                    child1, child2 = self.crossover(temp[0], temp[1])
                    children.append(child1)
                    children.append(child2)
                    if crossover_method == CrossoverMethod.Sequential:
                        temp = []
                    elif crossover_method == CrossoverMethod.Iterative:
                        temp = [temp[1]]
                if index >= len(parents):
                    index = 0
                    random.shuffle(parents)

        elif crossover_method == CrossoverMethod.Random:
            while num_children > len(children):
                child1, child2 = self.crossover(random.choice(parents), random.choice(parents))
                children.append(child1)
                children.append(child2)

        return children[:num_children]

    def mutate(self, solution: list[int]) -> list[int]:
        """
        Mutate a solution. Randomly change a value in the solution
        :param solution:
        :return:
        """
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, self.solution_length - 1)
            solution[mutation_point] = random.randint(0, self.solution_type[mutation_point])
        return solution

    def mutate_population(self, population: list[list[int]]) -> list[list[int]]:
        """
        Mutate the population
        :param population:
        :return:
        """
        return [self.mutate(solution) for solution in population]

    def log_population(self, population: list[list[int]], fitness: list[float] | None = None):
        """
        Print the population
        :return:
        """
        fitness = self.calculate_all_fitness() if fitness is None else fitness

        pop_fit = list(zip(population, fitness))
        pop_fit.sort(key=lambda x: x[1], reverse=True)

        for i, (solution, fit) in enumerate(pop_fit):
            for j, value in enumerate(solution):
                logger.debug(f"{value}:>3d")
                if j < len(solution) - 1:
                    logger.debug("-> ")
            logger.debug(f"Fitness: {fit:.2f}\n")

    def add_history(self, step: int, population: list[list[int]], fitness: list[float]):
        """
        Add the population and fitness to the history
        :param step:
        :param population:
        :param fitness:
        :return:
        """
        best_solution_index = max(range(len(fitness)), key=lambda i: fitness[i])
        avg_fitness = sum(fitness) / len(fitness)
        self.history.append({'step': step,
                             'population': list(population),
                             'fitness': list(fitness),
                             'best_solution': population[best_solution_index],
                             'best_fitness': fitness[best_solution_index],
                             'avg_fitness': avg_fitness})

    def plot_history(self, save_path=None):
        """
        Plot the history of the genetic algorithm and save the plot if save_path is provided
        :param save_path: path to save the plot
        :return:
        """
        plt.figure(figsize=(12, 6))
        plt.plot([step['step'] for step in self.history], [step['best_fitness'] for step in self.history], label='Best')
        plt.plot([step['step'] for step in self.history], [step['avg_fitness'] for step in self.history], label='Average')
        for i in range(len(self.history[0]['fitness'])):
            plt.scatter([step['step'] for step in self.history], [step['fitness'][i] for step in self.history], s=5)
        plt.title('Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def solve(self, max_generations: int = 100) -> list[int]:
        """
        Solve the problem using the genetic algorithm
        :param max_generations: maximum number of generations
        :return:
        """
        self.init()
        self.history = []

        for step in range(max_generations):
            fitness = self.calculate_all_fitness()

            self.add_history(step, self.population, fitness)

            # get the best solutions from the current generation
            elite = self.select_elites(self.num_elitism, fitness=fitness)
            # make a new generation
            parents = self.select_parents(self.num_parents, self.parent_select_method, tournament_size=5,
                                          fitness=fitness)

            num_children = self.population_size - len(elite)
            children = self.crossover_population(parents, self.crossover_method, num_children)
            children = self.mutate_population(children)

            # replace the population with the new generation
            self.population = elite + children

            # immigrate new solutions
            new_solutions = self.generate_population(self.population_size - len(self.population))
            self.population += new_solutions
            # logger.info(f"Generation {step + 1} completed with best fitness: {max(fitness)}")

        fitness = self.calculate_all_fitness()
        best_solution_index = max(range(len(fitness)), key=lambda i: fitness[i])
        return self.population[best_solution_index]