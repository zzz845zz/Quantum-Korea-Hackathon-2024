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
LayoutPasses = [
    None,
    TrivialLayout,
    DenseLayout,
    SabreLayout,
    CSPLayout,
    VF2Layout,
    # ApplyLayout,
    SabrePreLayout,
    # FullAncillaAllocation,
    # Layout2qDistance,
    # EnlargeWithAncilla,
]
RoutingPasses = [
    None,
    BasicSwap,
    LookaheadSwap,
    StochasticSwap,
    SabreSwap,
]
OptimizationPasses = [
    None,
    Collect1qRuns,
    Collect2qBlocks,
    Optimize1qGates,
    CollectMultiQBlocks,
    # Optimize1qGatesDecomposition,
    CXCancellation,
    # InverseCancellation,
    # CommutationAnalysis,
    CommutativeCancellation,
    # Optimize1qGatesSimpleCommutation,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    HoareOptimizer,
    # ElidePermutations,
    # Collect1qRuns,
    # Collect2qBlocks,
    # CollectMultiQBlocks,
    # CollectLinearFunctions,
]


class PassCombination:
    def __init__(self, LayoutPasses, RoutingPasses, OptimizationPasses):
        self.LayoutPasses: list = LayoutPasses
        self.RoutingPasses: list = RoutingPasses
        self.OptimizationPasses: list = OptimizationPasses

    def get_score(self, backend: BackendV2, base: StagedPassManager):
        cmap = backend.coupling_map
        pm = deepcopy(base)
        pm.layout += [lp(cmap) for lp in self.LayoutPasses if lp is not None]
        pm.routing += [rp(cmap) for rp in self.RoutingPasses if rp is not None]
        pm.optimization += [op() for op in self.OptimizationPasses if op is not None]

        tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler(
            [pm], backend, scorer(), num_qubits=np.arange(13, 14)
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

        # Empty individual
        population.append(
            PassCombination(
                LayoutPasses=[None] * self.gene_length,
                RoutingPasses=[None] * self.gene_length,
                OptimizationPasses=[None] * self.gene_length,
            )
        )

        # Randomly generate the first individual
        for _ in range(self.population_size - 1):
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
        return random.choices(self.population, weights=weights, k=2)

    def select_top_k(self, k):
        weights = [self.evaluate_fitness(individual) for individual in self.population]
        return [self.population[i] for i in np.argsort(weights)[-k:]]

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

        best_individual = max(self.population, key=self.evaluate_fitness)
        print(f"[gen] Init Best Fitness = {self.evaluate_fitness(best_individual)}")
        print("Layout Passes:")
        for lp in best_individual.LayoutPasses:
            print(lp)
        print("Routing Passes:")
        for rp in best_individual.RoutingPasses:
            print(rp)
        print("Optimization Passes:")
        for op in best_individual.OptimizationPasses:
            print(op)

        for generation in range(self.generations):
            print(f"[gen] Generation {generation}")
            new_population = []

            weights = [
                self.evaluate_fitness(individual) for individual in self.population
            ]
            top1, top2 = [self.population[i] for i in np.argsort(weights)[-2:]]
            new_population.extend([top1, top2])
            for _ in range((self.population_size // 2) - 1):
                print("[gen] Selecting parents")
                parent1, parent2 = random.choices(self.population, weights=weights, k=2)
                # parent1, parent2 = self.select()

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
