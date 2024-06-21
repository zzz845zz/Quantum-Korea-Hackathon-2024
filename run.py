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

seed = 10000


def append_transpiler(pm, name):
    transpiler_list.append(pm)
    transpiler_names.append(name)


pm_lv2 = generate_preset_pass_manager(
    backend=backend, optimization_level=2, seed_transpiler=seed
)
append_transpiler(
    pm=pm_lv2,
    name="preset pass manager level 2",
)
# transpiler_list.append(pm_lv2)
# transpiler_names.append("preset pass manager level 2")

append_transpiler(
    generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        layout_method="sabre",
        routing_method="bsabre",
        translation_method="synthesis",
    ),
    name="bsabre",
)

append_transpiler(
    StagedPassManager(
        stages=["init", "layout", "routing", "translation"],
        # init=pm_lv2.init,
        init=pm_lv2.init,
        # layout=pm_lv2.layout,
        layout=pm_lv2.layout,
        # layout="vf2",
        routing=pm_lv2.routing,
        # routing=my_routing,
        translation=pm_lv2.translation,
    ),
    name="pm_our",
)

# grade_transpiler(transpiler_list, backend, scorer)
transpiler_list = [
    generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        layout_method="sabre",
        routing_method="bsabre",
        translation_method="synthesis",
    )
]

num_qubits = np.arange(2, 7)
circuit = QuantumCircuit(5)
circuit.h(range(5))
isa_circuit = transpiler_list[0].run(circuit)
print((sum(isa_circuit.count_ops().values())))
