

from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit import Instruction, Parameter

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def check_connections_backend(backend): 
    instruc = backend.instructions

    qubit_tuple_sx = []
    for i in range(len(instruc)):
        if instruc[i][0].name == 'sx':
            qubit_tuple = instruc[i][1]
            

            qubit_tuple_sx.append(qubit_tuple) 

    qubit_tuple_x = []
    for i in range(len(instruc)):
        if instruc[i][0].name == 'x':
            qubit_tuple = instruc[i][1]
            

            qubit_tuple_x.append(qubit_tuple)        


    qubit_tuple_ecr = []
    for i in range(len(instruc)):
        if instruc[i][0].name == 'ecr':
            qubit_tuple = instruc[i][1]
            

            qubit_tuple_ecr.append(qubit_tuple) 

    qubit_tuple_rz = []
    for i in range(len(instruc)):
        if instruc[i][0].name == 'rz':
            qubit_tuple = instruc[i][1]
            

            qubit_tuple_rz.append(qubit_tuple) 


    return qubit_tuple_rz, qubit_tuple_ecr, qubit_tuple_sx, qubit_tuple_x
   


def check_connections_circuit(circuit):    
    qubit_tuple_ecr = []

    for i in range(len(circuit.data)):
        if circuit.data[i].operation.name == 'ecr':
            qubit_tuple = ()
            for j in range(len(circuit.data[i].qubits)):
                qubit_index = circuit.data[i].qubits[j]._index
                qubit_tuple = qubit_tuple + (qubit_index,)

            qubit_tuple_ecr.append(qubit_tuple)

    qubit_tuple_rz = []

    for i in range(len(circuit.data)):
        if circuit.data[i].operation.name == 'rz':
            qubit_tuple = ()
            for j in range(len(circuit.data[i].qubits)):
                qubit_index = circuit.data[i].qubits[j]._index
                qubit_tuple = qubit_tuple + (qubit_index,)

            qubit_tuple_rz.append(qubit_tuple)


    qubit_tuple_x = []

    for i in range(len(circuit.data)):
        if circuit.data[i].operation.name == 'x':
            qubit_tuple = ()
            for j in range(len(circuit.data[i].qubits)):
                qubit_index = circuit.data[i].qubits[j]._index
                qubit_tuple = qubit_tuple + (qubit_index,)

            qubit_tuple_x.append(qubit_tuple)

    qubit_tuple_sx = []

    for i in range(len(circuit.data)):
        if circuit.data[i].operation.name == 'sx':
            qubit_tuple = ()
            for j in range(len(circuit.data[i].qubits)):
                qubit_index = circuit.data[i].qubits[j]._index
                qubit_tuple = qubit_tuple + (qubit_index,)

            qubit_tuple_sx.append(qubit_tuple)

    return qubit_tuple_rz, qubit_tuple_ecr, qubit_tuple_sx, qubit_tuple_x


        
    

            

            
