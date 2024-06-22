from copy import deepcopy
import random
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_transpiler_service.transpiler_service import TranspilerService
from meta_sabre.meta_sabre_swap import MetaSabreSwap
from qkh2024.grader import scorer
from qiskit.transpiler.passes import *
from qiskit.transpiler.passmanager import StagedPassManager, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from utils import print_passes, grade_transpiler
from qiskit.providers import BackendV2


# Define possible passes
LayoutPasses = [TrivialLayout, DenseLayout, SabreLayout]
RoutingPasses = [BasicSwap, LookaheadSwap, StochasticSwap]
OptimizationPasses = [Optimize1qGates, Optimize1qGatesDecomposition, CXCancellation]


class PassCombination:
    def __init__(self, LayoutPasses, RoutingPasses, OptimizationPasses):
        self.LayoutPasses: list = LayoutPasses
        self.RoutingPasses: list = RoutingPasses
        self.OptimizationPasses: list = OptimizationPasses

    def get_score(self, backend: BackendV2, base: StagedPassManager):
        cmap = backend.coupling_map
        # print(cmap)
        pm = StagedPassManager(
            stages=[
                "init",
                "layout",
                "routing",
                "optimization",
                "translation",
                "scheduling",
            ],
            init=deepcopy(base.init),
            layout=deepcopy(base.layout) + [lp(cmap) for lp in self.LayoutPasses],
            routing=deepcopy(base.routing) + [rp(cmap) for rp in self.RoutingPasses],
            translation=deepcopy(base.translation),
            optimization=deepcopy(base.optimization)
            + [op() for op in self.OptimizationPasses],
            scheduling=deepcopy(base.scheduling),
        )

        tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler(
            [pm], backend, scorer(), num_qubits=np.arange(7, 8)
        )
        return tr_scores[0][0]


class GeneticAlgorithmCompilerOptimization:
    def __init__(
        self,
        population_size: int,
        gene_length: int,
        mutation_rate: float,
        crossover_rate: float,
        generations: int,
        # fitness_function,
        backend,
    ):
        """_summary_

        Args:
            population_size (int): Number of individuals in the population
            gene_length (int): Number of possible compiler passes
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
            generations (int): Number of generations
            fitness_function (function): Function to evaluate the fitness of an individual
            backend (_type_): IBM Quantum backend
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        # self.fitness_function = fitness_function
        self.backend = backend
        self.population = self.initialize_population()

        self.pm_lv2: StagedPassManager = generate_preset_pass_manager(
            backend=backend, optimization_level=2, seed_transpiler=10000
        )

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = PassCombination(
                LayoutPasses=random.choices(LayoutPasses, k=self.gene_length),
                RoutingPasses=random.choices(RoutingPasses, k=self.gene_length),
                OptimizationPasses=random.choices(
                    OptimizationPasses, k=self.gene_length
                ),
            )
            population.append(individual)
        return population

    def evaluate_fitness(self, individual: PassCombination):
        return individual.get_score(self.backend, base=self.pm_lv2)
        # return self.fitness_function(individual, self.backend)

    def select(self):
        weights = [self.evaluate_fitness(individual) for individual in self.population]
        print(weights)
        return random.choices(self.population, weights=weights, k=2)

    def crossover(self, parent1: PassCombination, parent2: PassCombination):
        # Single-point crossover
        if random.random() < self.crossover_rate:
            # Randomly select a crossover point
            point = random.randint(1, self.gene_length - 1)
            child1_layout = parent1.LayoutPasses[:point] + parent2.LayoutPasses[point:]
            child2_layout = parent2.LayoutPasses[:point] + parent1.LayoutPasses[point:]

            child1_routing = (
                parent1.RoutingPasses[:point] + parent2.RoutingPasses[point:]
            )
            child2_routing = (
                parent2.RoutingPasses[:point] + parent1.RoutingPasses[point:]
            )

            child1_optimization = (
                parent1.OptimizationPasses[:point] + parent2.OptimizationPasses[point:]
            )
            child2_optimization = (
                parent2.OptimizationPasses[:point] + parent1.OptimizationPasses[point:]
            )
        else:
            child1_layout, child2_layout = parent1.LayoutPasses, parent2.LayoutPasses
            child1_routing, child2_routing = (
                parent1.RoutingPasses,
                parent2.RoutingPasses,
            )
            child1_optimization, child2_optimization = (
                parent1.OptimizationPasses,
                parent2.OptimizationPasses,
            )

        child1 = PassCombination(child1_layout, child1_routing, child1_optimization)
        child2 = PassCombination(child2_layout, child2_routing, child2_optimization)
        return child1, child2

    def mutate(self, individual):
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                individual.LayoutPasses[i] = random.choice(LayoutPasses)
            if random.random() < self.mutation_rate:
                individual.RoutingPasses[i] = random.choice(RoutingPasses)
            if random.random() < self.mutation_rate:
                individual.OptimizationPasses[i] = random.choice(OptimizationPasses)
        return individual

    def run(self):
        for generation in range(self.generations):
            print(f"[gen] Generation {generation}")
            new_population = []
            for _ in range(self.population_size // 2):
                print("[gen] Selecting parents")
                parent1, parent2 = self.select()

                print("[gen] Crossover")
                child1, child2 = self.crossover(parent1, parent2)

                print("[gen] Mutating")
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            self.population = new_population
            best_individual = max(self.population, key=self.evaluate_fitness)
            print(
                f"[gen] Generation {generation}: Best Fitness = {self.evaluate_fitness(best_individual)}"
            )

        best_individual = max(self.population, key=self.evaluate_fitness)
        return best_individual, self.evaluate_fitness(best_individual)