from copy import deepcopy
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_transpiler_service.transpiler_service import TranspilerService
from genetic import GeneticAlgorithmCompilerOptimization
from meta_sabre.meta_sabre_swap import MetaSabreSwap
from qkh2024.grader import scorer
from qiskit.transpiler.passes import *
from qiskit.transpiler.passmanager import StagedPassManager, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from utils import print_passes, grade_transpiler

if __name__ == "__main__":
    # Connect to the Qiskit Runtime Service
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
    )
    backend = service.backend("ibm_sherbrooke")
    coupling_map = backend.coupling_map

    pm_lv2 = generate_preset_pass_manager(
        backend=backend, optimization_level=2, seed_transpiler=10000
    )

    # Define parameters
    population_size = 10
    gene_length = 5  # Number of passes in each category
    mutation_rate = 0.10
    crossover_rate = 0.7
    generations = 3

    # Instantiate and run the genetic algorithm
    ga = GeneticAlgorithmCompilerOptimization(
        population_size=population_size,
        gene_length=gene_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        generations=generations,
        # fitness_function=compiler_pass_fitness,
        backend=backend,
    )

    best_individual, best_fitness = ga.run()
    print(
        f"Best Compiler Pass Combination: {best_individual}, Best Fitness: {best_fitness}"
    )

    pm_ga = StagedPassManager(
        stages=[
            "init",
            "layout",
            "routing",
            "optimization",
            "translation",
            "scheduling",
        ],
        init=deepcopy(pm_lv2.init),
        layout=deepcopy(pm_lv2.layout)
        + [lp(coupling_map) for lp in best_individual.LayoutPasses],
        routing=deepcopy(pm_lv2.routing)
        + [rp(coupling_map) for rp in best_individual.RoutingPasses],
        translation=deepcopy(pm_lv2.translation),
        optimization=deepcopy(pm_lv2.optimization)
        + [op() for op in best_individual.OptimizationPasses],
        scheduling=deepcopy(pm_lv2.scheduling),
    )

    # passaa = pm_lv3.optimization

    # Grades
    scorer = scorer()
    transpiler_list = [
        # pm_msabre,
        pm_lv2,
        pm_ga,
        # pm_lv3,
        # pm_test,
    ]
    tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler(
        transpiler_list, backend, scorer, num_qubits=np.arange(7, 8)
    )

    for i in range(len(tr_scores)):
        print(f"Score for {i}: {tr_scores[i]}")
