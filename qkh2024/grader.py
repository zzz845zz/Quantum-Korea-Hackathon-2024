from typing import *
from colorama import Fore
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.compiler import transpile
from qiskit.quantum_info import hellinger_distance, hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict


class scorer:
    def __init__(
        self, 
    ):
        return

    def score(
        self, 
        circ: QuantumCircuit, 
        backend, 
    ):
        """
        A custom cost function that includes T1 and T2 computed during idle periods

        Parameters:
            circ (QuantumCircuit): circuit of interest
            layouts (list of lists): List of specified layouts
            backend (IBMQBackend): An IBM Quantum backend instance

        Returns:
            score(float): estimated fidelity as a score
        """

        fid = 1

        touched = set()
        dt = backend.dt
        num_qubits = backend.num_qubits
        
        t1s = [backend.qubit_properties(qq).t1 for qq in range(num_qubits)]
        t2s = [backend.qubit_properties(qq).t2 for qq in range(num_qubits)]

        for item in circ._data:
            for gate in backend.operation_names:
                if item[0].name == gate:
                    if (item[0].name == 'cz') or (item[0].name == 'ecr'):
                        q0 = circ.find_bit(item[1][0]).index
                        q1 = circ.find_bit(item[1][1]).index
                        fid *= 1 - backend.target[item[0].name][(q0, q1)].error
                        touched.add(q0)
                        touched.add(q1)
                    elif item[0].name == 'measure':
                        q0 = circ.find_bit(item[1][0]).index
                        fid *= 1 - backend.target[item[0].name][(q0, )].error
                        touched.add(q0)

                    elif item[0].name == 'delay':
                        q0 = circ.find_bit(item[1][0]).index
                        # Ignore delays that occur before gates
                        # This assumes you are in ground state and errors
                        # do not occur.
                        if q0 in touched:
                            time = item[0].duration * dt
                            fid *= 1-self._qubit_error(time, t1s[q0], t2s[q0])
                    else:
                        q0 = circ.find_bit(item[1][0]).index
                        fid *= 1 - backend.target[item[0].name][(q0, )].error
                        touched.add(q0)

        return fid

    def validate(
        self, 
        circuit_target: QuantumCircuit, 
        circuit_transpiled: QuantumCircuit, 
        backend, 
    ):
        self.grader_operation_check(circuit_transpiled, backend)
        self.grader_connection_check(circuit_transpiled, backend)
        self.grader_circuit_accuracy(circuit_target, circuit_transpiled)

    def grader_operation_check(
        self, 
        circ: QuantumCircuit, 
        backend, 
    ):
        """
        Takes a backend and a circuit and see whether the gates in the circuit are supported by the backend

        Parameters : 
            backend: Target backend
            circuit (QuantumCircuit): Circuit whose gates are to be checked with the gates in the target backend
        """

        op_names_backend : list[str] = backend.operation_names
        op_names_circuit_dict : dict[str,int]  = circ.count_ops().items()
        #use set.
        for keys, _ in op_names_circuit_dict:
            if keys in op_names_backend:
                None
            else : 
                print(Fore.RED + f'The backend does not support {keys} gate')
        return print(Fore.GREEN + 'Basis gate check passed!')

    def grader_connection_check(
        self, 
        circ: QuantumCircuit, 
        backend, 
    ):
        """"
        Takes in a Quantum Circuit and a Backend and checks whether the conections in the circuit matches 
        with the topology of the backend. 

        Parameters: 
            backend: Target backend in which the circuit must fit.
            circuit (QuantumCircuit): Circuit whose connections are to be matched with the backend

        Returns: Message: Whether the circuit fits in the topology of the backend. 
        """

        conn_backend = self._connections_backend(backend)
        conn_circ = self._connections_circuit(circ)

        gate_conn_set_backend : dict[str,set] = {}
        for keys,items in conn_backend.items():
            gate_conn_set_backend[keys] = set(items)

        gate_conn_set_circuit : dict[str,set] = {}
        for keys,items in conn_circ.items():
            gate_conn_set_circuit[keys] = set(items)

        check_connections : list[int] = []
        for keys, items in gate_conn_set_circuit.items():
            check_connections.append(int(gate_conn_set_circuit[keys].issubset(gate_conn_set_backend[keys])))

        if 0 in check_connections:
            print(Fore.RED + 'The connections does not match with the backend')

        else : 
            print(Fore.GREEN + 'The connections match with the backend')

    def grader_circuit_accuracy(
        self, 
        circuit_target: QuantumCircuit, 
        circuit_transpiled: QuantumCircuit, 
        simulator = AerSimulator(method="statevector"), 
        ε: float = 0.75, 
    ):
        """
        Takes the test circuit and compares it with the target circuit.
        It computes the Hellinger distance and Hellinger Fidelity of the probability 
        distribution of the outputs of the to circuits and checks whether they are in
        the tolerance level.

        Parameters : 
            circuit_target (QuantumCircuit): Original Quantum Circuit
            circuit_transpiled (QuantumCircuit): Transpiled Circuit.
            simulator: default = AerSimulator
            epsilon: Tolerance. default = 0.75

        Output :
            Prints out whether the transpiled circuit is within epsilon accuracy of the original circuit.
        """

        circuit_target = transpile(self._remove_idle_qwires(circuit_target), backend = simulator)
        circuit_transpiled = transpile(self._remove_idle_qwires(circuit_transpiled), backend = simulator)

        counts_target = self._counts(circuit_target)
        counts_transpiled = self._counts(circuit_transpiled)

        hell_dist : float = hellinger_distance(counts_target, counts_transpiled)
        hell_fid : float = hellinger_fidelity(counts_target, counts_transpiled)

        if hell_fid >= ε or hell_dist <= 1-ε:
            print(Fore.GREEN + 'Congratualtions! Your circuit is within the given tolerance of the original circuit')

        else :
            print(Fore.RED + 'Oops! Your circuit is not within the given tolerance of the original circuit\n Try again!')

    def _qubit_error(
        self, 
        time: float, 
        t1: float, 
        t2: float, 
    ) -> float:
        """
        Compute the approx. idle error from T1 and T2
        Parameters:
            time (float): Delay time in sec
            t1 (float): T1 time in sec
            t2 (float): T2 time in sec
        Returns:
            float: Idle error
        """

        t2 = min(t1, t2)
        rate1 = 1/t1
        rate2 = 1/t2
        p_reset = 1-np.exp(-time*rate1)
        p_z = (1-p_reset)*(1-np.exp(-time*(rate2-rate1)))/2
        return p_z + p_reset

    def _connections_backend(
        self, 
        backend, 
    ) -> dict[str, list]: 
        """
        Takes a backend and returns the connections.

        Parameters : 
            Inputs a backend

        Returns: 
            The connection of the gates as a dictionary,
            the keys as gates and values as connections
        """

        instruc: list = backend.instructions
        operations: list = backend.operations
        oper_key : list[str] = []

        for items in operations:
            if isinstance(items, Instruction) and items.num_clbits == 0 :
                oper_key.append(items.name)

        list_dict : dict[str, list] = {name : [] for name in oper_key}
        for gates in range(len(oper_key)):
            for i in range(len(instruc)):
                if instruc[i][0].name == oper_key[gates]:
                    qubit_tuple = instruc[i][1]
                    list_dict[oper_key[gates]].append(qubit_tuple)

        return list_dict

    def _connections_circuit(
        self, 
        circ : QuantumCircuit
    ) -> dict[str,list]: 
        """
        Takes a Quantum Circuit and returns the connections.

        Parameters : 
            Inputs a QuantumCircuit

        Returns: 
            The connection of the gates as a dictionary,
            the keys as gates and values as connections
        """

        gates_used: list = []
        for keys, _ in circ.count_ops().items():
            gates_used.append(keys)
        list_dict : dict[str,list] = {name : [] for name in gates_used}
        for gates in gates_used: 
            for i in range(len(circ.data)):
                if circ.data[i].operation.name == gates:
                    qubit_tuple = ()
                    for j in range(len(circ.data[i].qubits)):
                        qubit_index = circ.data[i].qubits[j]._index
                        qubit_tuple = qubit_tuple + (qubit_index,)
                    list_dict[gates].append(qubit_tuple)   
        return list_dict

    def _counts(
        self, 
        circ : QuantumCircuit, 
        simulator = AerSimulator(method="statevector"), 
    ) -> dict[str, int]:
        """
        Function to add measurements to a provided circuit and run it on the AerSimulator.

        Parameters 
            circuit (QuantumCircuit)
            simulator: default = AerSimulator

        Returns  
            counts in a dict.
        """

        if 'measure' in circ.count_ops():
            None
        else : 
            circ.measure_active()
        result = simulator.run(circ, backend = simulator).result()
        counts = result.get_counts()

        return counts

    def _remove_idle_qwires(
        self, 
        circ: QuantumCircuit, 
    ):
        dag = circuit_to_dag(circ)

        idle_wires = list(dag.idle_wires())
        for w in idle_wires:
            dag._remove_idle_wire(w)
            dag.qubits.remove(w)

        dag.qregs = OrderedDict()

        return dag_to_circuit(dag)
