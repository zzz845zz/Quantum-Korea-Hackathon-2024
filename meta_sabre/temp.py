import logging
from copy import deepcopy
from itertools import cycle
import numpy as np

from qiskit.dagcircuit import DAGCircuit
from legacy.dagcircuit import DAGNode
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

# from qiskit.dagcircuit import DAGOpNode

# For debugging
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.circuit.library import QFT

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay cooefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class MetaSabreSwap(TransformationPass):
    def __init__(self, coupling_map, heuristic="basic", seed=None, max_depth=2):
        super().__init__()
        self.coupling_map = coupling_map
        self.heuristic = heuristic
        self.seed = seed
        self.applied_gates = None
        self.qubits_decay = None
        self.max_depth = max_depth

    def run_suppl(
        self, depth, dag, rng, front_layer, current_layout, mapped_dag, num_search_steps
    ):
        """Supplementary function to run() that allows recursion with depth parameter.

        Args:
            depth (int): Depth of recursion.
            dag (DAGCircuit): Input circuit.
            rng (np.random.Generator): Random number generator.
            front_layer (list): List of nodes to be executed. (deepcopy needed?)
            current_layout (Layout): Current layout of qubits. (deepcopy needed?)
            mapped_dag (DAGCircuit): Output circuit. (deepcopy needed?)
            num_search_steps (int): Number of search steps taken so far. (deepcopy needed?)
        Returns:
            mapped_dag (DAGCircuit): Output circuit if depth is 0.
            score (int): Score of the layout if depth is not 0.
        """
        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    physical_qubits = (current_layout[v0], current_layout[v1])
                    if physical_qubits in self.coupling_map.get_edges():
                        execute_gate_list.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    new_node = _transform_gate_for_layout(node, current_layout)
                    mapped_dag.apply_operation_back(
                        new_node.op,
                        new_node.qargs,
                        new_node.cargs,
                    )
                    front_layer.remove(node)
                    self.applied_gates.add(node)
                    for successor in dag.quantum_successors(node):
                        if isinstance(successor, DAGOpNode):
                            continue
                        if self._is_resolved(successor, dag):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()

                if depth == 0:
                    # Diagnostics
                    logger.debug("free! %s", [str(n) for n in execute_gate_list])
                    logger.debug("front_layer: %s", [str(n) for n in front_layer])
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            extended_set = self._obtain_extended_set(dag, front_layer)
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            swap_scores = dict.fromkeys(swap_candidates, 0)
            for swap_qubits in swap_scores:
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                if depth < self.max_depth:
                    score = self.run_suppl(
                        depth + 1,
                        dag,
                        rng,
                        deepcopy(front_layer),
                        deepcopy(trial_layout),
                        deepcopy(mapped_dag),
                        deepcopy(num_search_steps),
                    )
                else:
                    score = self._score_heuristic(
                        self.heuristic,
                        front_layer,
                        extended_set,
                        trial_layout,
                        swap_qubits,
                    )
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())

            if depth > 0:
                return min_score

            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (x[0].index, x[1].index))
            best_swap = rng.choice(best_swaps)
            swap_node = DAGOpNode(op=SwapGate(), qargs=best_swap)
            swap_node = _transform_gate_for_layout(swap_node, current_layout)
            mapped_dag.apply_operation_back(swap_node.op, swap_node.qargs)
            current_layout.swap(*best_swap)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            # Diagnostics
            logger.debug("SWAP Selection...")
            logger.debug("extended_set: %s", [(n.name, n.qargs) for n in extended_set])
            logger.debug("swap scores: %s", swap_scores)
            logger.debug("best swap: %s", best_swap)
            logger.debug("qubits decay: %s", self.qubits_decay)

        self.property_set["final_layout"] = current_layout

        return mapped_dag

    def run(self, dag):
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = _copy_circuit_metadata(dag)

        # Assume bidirectional couplings, fixing gate direction is easy later.
        self.coupling_map.make_symmetric()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = {qubit: 1 for qubit in dag.qubits}

        # Start algorithm from the front layer and iterate until all gates done.
        num_search_steps = 0
        front_layer = dag.front_layer()
        self.applied_gates = set()

        return self.run_suppl(
            0, dag, rng, front_layer, current_layout, mapped_dag, num_search_steps
        )

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _is_resolved(self, node, dag):
        """Return True if all of a node's predecessors in dag are applied."""
        predecessors = dag.quantum_predecessors(node)
        predecessors = filter(lambda x: isinstance(x, DAGOpNode), predecessors)
        return all([n in self.applied_gates for n in predecessors])

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        # TODO: use layers instead of bfs_successors so long range successors aren't included.
        extended_set = set()
        bfs_successors_pernode = [dag.bfs_successors(n) for n in front_layer]
        node_lookahead_exhausted = [False] * len(front_layer)
        for i, node_successor_generator in cycle(enumerate(bfs_successors_pernode)):
            if all(node_lookahead_exhausted) or len(extended_set) >= EXTENDED_SET_SIZE:
                break

            try:
                _, successors = next(node_successor_generator)
                successors = list(
                    filter(lambda x: x.type == "op" and len(x.qargs) == 2, successors)
                )
            except StopIteration:
                node_lookahead_exhausted[i] = True
                continue

            successors = iter(successors)
            while len(extended_set) < EXTENDED_SET_SIZE:
                try:
                    extended_set.add(next(successors))
                except StopIteration:
                    break

        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted(
                        [virtual, virtual_neighbor],
                        key=lambda q: (q.register.name, q.index),
                    )
                    candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _score_heuristic(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        if heuristic == "basic":
            return sum(
                self.coupling_map.distance(*[layout[q] for q in node.qargs])
                for node in front_layer
            )

        elif heuristic == "lookahead":
            first_cost = self._score_heuristic("basic", front_layer, [], layout)
            first_cost /= len(front_layer)

            second_cost = self._score_heuristic("basic", extended_set, [], layout)
            second_cost = 0.0 if not extended_set else second_cost / len(extended_set)

            return first_cost + EXTENDED_SET_WEIGHT * second_cost

        elif heuristic == "decay":
            return max(
                self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
            ) * self._score_heuristic("lookahead", front_layer, extended_set, layout)

        else:
            raise TranspilerError("Heuristic %s not recognized." % heuristic)


def _copy_circuit_metadata(source_dag):
    """Return a copy of source_dag with metadata but empty."""
    target_dag = DAGCircuit()
    target_dag.name = source_dag.name

    for qreg in source_dag.qregs.values():
        target_dag.add_qreg(qreg)
    for creg in source_dag.cregs.values():
        target_dag.add_creg(creg)

    return target_dag


def _transform_gate_for_layout(op_node, layout):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = deepcopy(op_node)

    device_qreg = op_node.qargs[0]._register
    premap_qargs = op_node.qargs
    mapped_qargs = map(lambda x: device_qreg[layout[x]], premap_qargs)
    mapped_op_node.qargs = mapped_op_node.op.qargs = list(mapped_qargs)

    return mapped_op_node


if __name__ == "__main__":
    pm = PassManager()
    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    pm.append(MetaSabreSwap(coupling_map, heuristic="decay", seed=42, max_depth=2))

    circuit = QFT(4)
    res = pm.run(circuit)

    print(circuit)
    print(res)
