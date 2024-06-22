from colorama import Fore
import numpy as np
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passmanager import PassManager, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.circuit.random import random_circuit


def print_passes(target: StagedPassManager, stage: str = None):
    if stage in [None, "init"]:
        print("----- init stage ")
        # print(list_stage_plugins("init"))
        for task in target.init.to_flow_controller().tasks:
            print(" -", type(task).__name__)

    if stage in [None, "layout"]:
        print("----- layout stage ")
        # print(list_stage_plugins("layout"))
        for controller_group in target.layout.to_flow_controller().tasks:
            # print(controller_group)
            tasks = getattr(controller_group, "tasks", [])
            for task in tasks:
                print(" - ", str(type(task).__name__))

    if stage in [None, "routing"]:
        print("----- routing stage")
        for controller_group in target.routing.to_flow_controller().tasks:
            print(controller_group)
            tasks = getattr(controller_group, "tasks", [])
            for task in tasks:
                print(" - ", str(type(task).__name__))

    if stage in [None, "translation"]:
        print("----- translation stage")
        # print(list_stage_plugins("translation"))
        for controller_group in target.translation.to_flow_controller().tasks:
            # print(controller_group)
            tasks = getattr(controller_group, "tasks", [])
            for task in tasks:
                print(" - ", str(type(task).__name__))

    if stage in [None, "optimization"]:
        print("---- optimization stage")
        for controller_group in target.optimization.to_flow_controller().tasks:
            print(controller_group)
            # for task in controller_group.tasks:
            #     tasks = getattr(controller_group, "tasks", [])
            #     for task in tasks:
            #         print(" - ", str(type(task).__name__))


def grade_transpiler(transpiler_list, backend, scorer, num_qubits=None):
    tr_depths = [[] for i in range(len(transpiler_list))]
    tr_gate_counts = [[] for i in range(len(transpiler_list))]
    tr_cnot_counts = [[] for i in range(len(transpiler_list))]
    tr_scores = [[] for i in range(len(transpiler_list))]
    # print(tr_depths, len(transpiler_list))

    if num_qubits is None:
        num_qubits = np.arange(2, 15)

    for nq in num_qubits:
        print(f"Start transpiling the {nq}-qubit circuit")
        circuit = QuantumCircuit(nq)
        circuit.h(range(nq))
        circuit.append(QFT(nq, do_swaps=False, inverse=True).decompose(), range(nq))
        for i in range(len(transpiler_list)):
            isa_circuit = transpiler_list[i].run(circuit)
            scorer.validate(circuit, isa_circuit, backend)
            tr_depths[i].append(isa_circuit.depth())
            tr_gate_counts[i].append(sum(isa_circuit.count_ops().values()))
            tr_cnot_counts[i].append(isa_circuit.num_nonlocal_gates())
            tr_scores[i].append(scorer.score(isa_circuit, backend))

    return tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores


def grade_transpiler_circuit(transpiler_list, backend, scorer, circuit):
    tr_depths = [[] for i in range(len(transpiler_list))]
    tr_gate_counts = [[] for i in range(len(transpiler_list))]
    tr_cnot_counts = [[] for i in range(len(transpiler_list))]
    tr_scores = [[] for i in range(len(transpiler_list))]
    # print(tr_depths, len(transpiler_list))

    print(Fore.GREEN + f"Start transpiling the given circuit")
    for i in range(len(transpiler_list)):
        isa_circuit = transpiler_list[i].run(circuit)
        # scorer.validate(circuit, isa_circuit, backend)
        tr_depths[i].append(isa_circuit.depth())
        tr_gate_counts[i].append(sum(isa_circuit.count_ops().values()))
        tr_cnot_counts[i].append(isa_circuit.num_nonlocal_gates())
        tr_scores[i].append(scorer.score(isa_circuit, backend))

    return tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores


# def plot_

# import matplotlib.pyplot as plt

# ax = num_qubits
# opt_list = ['Optimization Level 1', 'Optimization Level 2', 'Optimization Level 3', 'AI Optimization']
# markers = ['o', 's', '^', '*']
# linestyles = ['-', '--', '-.', ':']
# colors = ['#FF6666', '#FFCC66', '#99FF99', '#66B2FF']

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

# # Plot 1: Circuit Depth
# for i in range(4):
#     ax1.plot(ax, tr_depths[i], label=opt_list[i], marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
# ax1.set_xlabel("Number of qubits", fontsize=12)
# ax1.set_ylabel("Depth", fontsize=12)
# ax1.set_title("Circuit Depth of Transpiled Circuit", fontsize=14)
# ax1.legend(fontsize=10)

# # Plot 2: Total Number of Gates
# for i in range(4):
#     ax2.plot(ax, tr_gate_counts[i], label=opt_list[i], marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
# ax2.set_xlabel("Number of qubits", fontsize=12)
# ax2.set_ylabel("# of Total Gates", fontsize=12)
# ax2.set_title("Total Number of Gates of Transpiled Circuit", fontsize=14)
# ax2.legend(fontsize=10)

# # Plot 3: Total Number of CNOTs
# for i in range(4):
#     ax3.plot(ax, tr_cnot_counts[i], label=opt_list[i], marker=markers[i],markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
# ax3.set_xlabel("Number of qubits", fontsize=12)
# ax3.set_ylabel("# of CNOTs", fontsize=12)
# ax3.set_title("Total Number of CNOTs of Transpiled Circuit", fontsize=14)
# ax3.legend(fontsize=10)

# # Plot 4: Score of Transpiled Circuit
# for i in range(4):
#     ax4.plot(ax, tr_scores[i], label=opt_list[i], marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
# ax4.set_xlabel("Number of qubits", fontsize=12)
# ax4.set_ylabel("Score of Transpiled Circuit", fontsize=12)
# ax4.set_title("Score of Transpiled Circuit", fontsize=14)
# ax4.legend(fontsize=10)

# fig.tight_layout()
# plt.show()
