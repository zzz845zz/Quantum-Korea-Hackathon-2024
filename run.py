import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService

# from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qkh2024.grader import scorer
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passmanager import PassManager, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from utils import print_passes, grade_transpiler

service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
)
backend = service.backend("ibm_sherbrooke")
coupling_map = backend.coupling_map

scorer = scorer()

transpiler_list = []
transpiler_names = []


def append_transpiler(pm, name):
    transpiler_list.append(pm)
    transpiler_names.append(name)


seed = 10000
pm_lv2 = generate_preset_pass_manager(
    backend=backend, optimization_level=2, seed_transpiler=seed
)
# append_transpiler(
#     pm=pm_lv2,
#     name="preset pass manager level 2",
# )
# transpiler_list.append(pm_lv2)
# transpiler_names.append("preset pass manager level 2")

# grade_transpiler(transpiler_list, backend, scorer)
transpiler_list = [
    pm_lv2,
    generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        # layout_method="sabre",
        routing_method="msabre",
        # translation_method="translate",
    ),
]
print_passes(transpiler_list[-1])
tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler(
    transpiler_list, backend, scorer, num_qubits=np.arange(5, 6)
)
# print(tr_depths)
for i in range(len(tr_scores)):
    print(f"Score for {i}: {tr_scores[i]}")
