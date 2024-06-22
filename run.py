from copy import deepcopy
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_transpiler_service.transpiler_service import TranspilerService
from genetic import GeneticAlgorithmCompilerOptimization
from meta_sabre.meta_sabre_swap import MetaSabreSwap
from qkh2024.grader import scorer
from qiskit.transpiler.passes import *
from qiskit.transpiler.passmanager import StagedPassManager, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from utils import (
    grade_transpiler_circuit,
    print_passes,
    grade_transpiler,
)
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT
import csv


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # circuit: "QFT" or "random"
    parser.add_argument(
        "-c",
        "--circuit",
        dest="circuit",
        choices=["qft", "random"],
        required=True,
        help="Circuit to transpile",
    )

    # Number of qubits
    parser.add_argument(
        "-n",
        "--num-qubits",
        dest="num_qubits",
        type=int,
        required=True,
        help="Number of qubits in the circuit",
    )

    # Output file
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default=None,
        required=True,
        help="Output file",
    )

    return parser.parse_args()


def get_pm(backend, optimization_level):
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level, seed_transpiler=10000
    )
    return pm


def get_pm_ga(
    population_size: int,
    gene_length: int,
    mutation_rate: float,
    crossover_rate: float,
    generations: int,
    backend,
    pm_base,
    circuit,
):
    pm_ga = GeneticAlgorithmCompilerOptimization(
        population_size=population_size,
        gene_length=gene_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        generations=generations,
        backend=backend,
        pm_base=pm_base,  # 이 pm으로부터 heuristic 시작
        circuit=circuit,  # 이 circuit을 타겟으로 점수 평가
    )
    best_individual, best_fitness, pm_ga = pm_ga.run()

    # Print the best individual
    print(
        f"Best Compiler Pass Combination: {best_individual}, Best Fitness: {best_fitness}"
    )
    for lp in best_individual.LayoutPasses:
        print(lp)
    for rp in best_individual.RoutingPasses:
        print(rp)
    for op in best_individual.OptimizationPasses:
        print(op)
    return pm_ga


if __name__ == "__main__":
    args = parse_args()

    # Connect to the Qiskit Runtime Service
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
    )
    backend = service.backend("ibm_sherbrooke")

    # Construct the circuit
    nq = args.num_qubits
    if args.circuit == "qft":
        circuit = QuantumCircuit(nq)
        circuit.h(range(nq))
        circuit.append(QFT(nq, do_swaps=False, inverse=True).decompose(), range(nq))
    elif args.circuit == "random":
        circuit = random_circuit(nq, nq + 1)
    else:
        raise ValueError("Invalid circuit type")
    print(circuit)

    # Define the pass managers
    # - Level 2 pass manager
    # - Genetic Algorithm pass manager
    pm_lv2 = get_pm(backend, optimization_level=2)
    pm_lv3 = get_pm(backend, optimization_level=3)
    pm_ga = get_pm_ga(
        population_size=10,  # Number of individuals
        gene_length=10,  # Number of passes in each category
        mutation_rate=0.1,
        crossover_rate=0.5,
        generations=5,  # Number of iterations
        backend=backend,
        pm_base=pm_lv2,
        circuit=circuit,
    )

    # Grades
    scorer = scorer()
    transpiler_list = [
        pm_lv2,
        pm_lv3,
        pm_ga,
    ]
    tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler_circuit(
        transpiler_list,
        backend,
        scorer,
        circuit=circuit,
    )
    for i in range(len(tr_scores)):
        print(f"Score for {i}: {tr_scores[i]}")

    # Save the results as csv
    # (circuit, nqubits, o2 result, o3 result, o2-ga result)
    with open(args.output, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [args.circuit, nq, tr_scores[0][0], tr_scores[1][0], tr_scores[2][0]]
        )
