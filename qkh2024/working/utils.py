

from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit import Instruction, Parameter

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager



def connections_backend(backend : object)->dict[str,list]: 
    """
    Takes a backend and returns the connections.

    Parameters : 
          Inputs a backend

    Returns: 
          The connection of the gates as a dictionary,
          the keys as gates and values as connections
    
    """
    instruc : list = backend.instructions
    operations : list= backend.operations
    oper_key : list[str] = []
    

    for items in backend.operations:
        if isinstance(items, Instruction) and items.num_clbits == 0 :
            oper_key.append(items.name)




    list_dict : dict[str, list] = {name : [] for name in oper_key}
    for gates in range(len(oper_key)):
        for i in range(len(instruc)):
            if instruc[i][0].name == oper_key[gates]:
                qubit_tuple = instruc[i][1]

                list_dict[oper_key[gates]].append(qubit_tuple)


    return list_dict

def connections_circuit(circuit : QuantumCircuit)-> dict[str,list]: 

    """
    Takes a Quantum Circuit and returns the connections.

    Parameters : 
          Inputs a Quantum Circuit

    Returns: 
          The connection of the gates as a dictionary,
          the keys as gates and values as connections
    
    """
    gates_used: list = []
    for keys, items in circuit.count_ops().items():
        gates_used.append(keys)
    list_dict : dict[str,list] = {name : [] for name in gates_used}
    for gates in gates_used: 
        for i in range(len(circuit.data)):
            if circuit.data[i].operation.name == gates:
                qubit_tuple = ()
                for j in range(len(circuit.data[i].qubits)):
                    qubit_index = circuit.data[i].qubits[j]._index
                    qubit_tuple = qubit_tuple + (qubit_index,)

                list_dict[gates].append(qubit_tuple)   
    
    return list_dict


        
    

            

            


# def check_connections_backend(backend): 
#     instruc = backend.instructions

#     qubit_tuple_sx = []
#     for i in range(len(instruc)):
#         if instruc[i][0].name == 'sx':
#             qubit_tuple = instruc[i][1]
            

#             qubit_tuple_sx.append(qubit_tuple) 

#     qubit_tuple_x = []
#     for i in range(len(instruc)):
#         if instruc[i][0].name == 'x':
#             qubit_tuple = instruc[i][1]
            

#             qubit_tuple_x.append(qubit_tuple)        


#     qubit_tuple_ecr = []
#     for i in range(len(instruc)):
#         if instruc[i][0].name == 'ecr':
#             qubit_tuple = instruc[i][1]
            

#             qubit_tuple_ecr.append(qubit_tuple) 

#     qubit_tuple_rz = []
#     for i in range(len(instruc)):
#         if instruc[i][0].name == 'rz':
#             qubit_tuple = instruc[i][1]
            

#             qubit_tuple_rz.append(qubit_tuple) 


#     return qubit_tuple_rz, qubit_tuple_ecr, qubit_tuple_sx, qubit_tuple_x
   


# def check_connections_circuit(circuit):    
#     qubit_tuple_ecr = []

#     for i in range(len(circuit.data)):
#         if circuit.data[i].operation.name == 'ecr':
#             qubit_tuple = ()
#             for j in range(len(circuit.data[i].qubits)):
#                 qubit_index = circuit.data[i].qubits[j]._index
#                 qubit_tuple = qubit_tuple + (qubit_index,)

#             qubit_tuple_ecr.append(qubit_tuple)

#     qubit_tuple_rz = []

#     for i in range(len(circuit.data)):
#         if circuit.data[i].operation.name == 'rz':
#             qubit_tuple = ()
#             for j in range(len(circuit.data[i].qubits)):
#                 qubit_index = circuit.data[i].qubits[j]._index
#                 qubit_tuple = qubit_tuple + (qubit_index,)

#             qubit_tuple_rz.append(qubit_tuple)


#     qubit_tuple_x = []

#     for i in range(len(circuit.data)):
#         if circuit.data[i].operation.name == 'x':
#             qubit_tuple = ()
#             for j in range(len(circuit.data[i].qubits)):
#                 qubit_index = circuit.data[i].qubits[j]._index
#                 qubit_tuple = qubit_tuple + (qubit_index,)

#             qubit_tuple_x.append(qubit_tuple)

#     qubit_tuple_sx = []

#     for i in range(len(circuit.data)):
#         if circuit.data[i].operation.name == 'sx':
#             qubit_tuple = ()
#             for j in range(len(circuit.data[i].qubits)):
#                 qubit_index = circuit.data[i].qubits[j]._index
#                 qubit_tuple = qubit_tuple + (qubit_index,)

#             qubit_tuple_sx.append(qubit_tuple)

#     return qubit_tuple_rz, qubit_tuple_ecr, qubit_tuple_sx, qubit_tuple_x


        
    

            

            
