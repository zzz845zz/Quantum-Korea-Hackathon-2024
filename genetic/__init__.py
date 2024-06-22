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
from utils import grade_transpiler_circuit, print_passes, grade_transpiler
from qiskit.providers import BackendV2
from colorama import Fore

# Define possible passes
LayoutPasses = [
    None,
    # TrivialLayout,
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
    # CollectLinearFunctions,
    CXCancellation,
    # CommutationAnalysis,
    CommutativeCancellation,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    HoareOptimizer,
    # ElidePermutations,
]
# InverseCancellation,
# Optimize1qGatesSimpleCommutation,
# Optimize1qGatesDecomposition,


class PassCombination:
    def __init__(self, LayoutPasses, RoutingPasses, OptimizationPasses):
        self.LayoutPasses: list = LayoutPasses
        self.RoutingPasses: list = RoutingPasses
        self.OptimizationPasses: list = OptimizationPasses

    def get_score(self, backend: BackendV2, base: StagedPassManager, circuit=None):
        cmap = backend.coupling_map
        pm = deepcopy(base)
        pm.layout += [lp(cmap) for lp in self.LayoutPasses if lp is not None]
        pm.routing += [rp(cmap) for rp in self.RoutingPasses if rp is not None]
        pm.optimization += [op() for op in self.OptimizationPasses if op is not None]

        tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler_circuit(
            [pm], backend, scorer(), circuit=circuit
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
        pm_base=None,
        circuit=None,
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
        self.circuit = circuit

        if pm_base is None:
            self.pm_base: StagedPassManager = generate_preset_pass_manager(
                backend=backend, optimization_level=2, seed_transpiler=10000
            )
        else:
            self.pm_base = pm_base

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
        # print("Layout Passes:")
        # for lp in individual.LayoutPasses:
        #     print(lp)
        # print("Routing Passes:")
        # for rp in individual.RoutingPasses:
        #     print(rp)
        # print("Optimization Passes:")
        # for op in individual.OptimizationPasses:
        #     print(op)

        return individual.get_score(
            self.backend, base=self.pm_base, circuit=self.circuit
        )

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
        print(
            Fore.BLUE
            + f"[gen] Init Best Fitness = {self.evaluate_fitness(best_individual)}"
        )
        # print(Fore.BLUE + "Layout Passes:")
        # for lp in best_individual.LayoutPasses:
        #     print(lp)
        # print(Fore.BLUE + "Routing Passes:")
        # for rp in best_individual.RoutingPasses:
        #     print(rp)
        # print(Fore.BLUE + "Optimization Passes:")
        # for op in best_individual.OptimizationPasses:
        #     print(op)

        for generation in range(self.generations):
            print(Fore.BLUE + f"[gen] Generation {generation}")
            new_population = []

            weights = [
                self.evaluate_fitness(individual) for individual in self.population
            ]
            top1, top2 = [self.population[i] for i in np.argsort(weights)[-2:]]
            new_population.extend([top1, top2])
            for _ in range((self.population_size // 2) - 1):
                print(Fore.BLUE + "[gen] Selecting parents")
                parent1, parent2 = random.choices(self.population, weights=weights, k=2)
                # parent1, parent2 = self.select()

                print(Fore.BLUE + "[gen] Crossover")
                child1, child2 = self.crossover(parent1, parent2)

                print(Fore.BLUE + "[gen] Mutating")
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            self.population = new_population
            best_individual = max(self.population, key=self.evaluate_fitness)
            print(
                Fore.BLUE
                + f"[gen] Generation {generation}: Best Fitness = {self.evaluate_fitness(best_individual)}"
            )

        best_individual = max(self.population, key=self.evaluate_fitness)

        cmap = self.backend.coupling_map
        best_pm = deepcopy(self.pm_base)
        best_pm.layout += [
            lp(cmap) for lp in best_individual.LayoutPasses if lp is not None
        ]
        best_pm.routing += [
            rp(cmap) for rp in best_individual.RoutingPasses if rp is not None
        ]
        best_pm.optimization += [
            op() for op in best_individual.OptimizationPasses if op is not None
        ]
        return best_individual, self.evaluate_fitness(best_individual), best_pm
