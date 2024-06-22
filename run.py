import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_transpiler_service.transpiler_service import TranspilerService
from meta_sabre.meta_sabre_swap import MetaSabreSwap
from qkh2024.grader import scorer
from qiskit.transpiler.passmanager import StagedPassManager, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from utils import print_passes, grade_transpiler

def get_routing_pm(coupling_map):
    # Get the MetaSabreSwap PassManager
    pm = PassManager()
    pm.append(
        MetaSabreSwap(
            coupling_map=coupling_map,
        )
    )

    return pm

if __name__ == "__main__":
    # Connect to the Qiskit Runtime Service
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
    )
    backend = service.backend("ibm_sherbrooke")
    coupling_map = backend.coupling_map

    # Create pass managers
    seed = 10000
    pm_lv2 = generate_preset_pass_manager(
        backend=backend, optimization_level=2, seed_transpiler=seed
    )

    pm_msabre = StagedPassManager(
    stages=['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling'],
        init=pm_lv2.init,
        layout=pm_lv2.layout,
        routing=PassManager(
                MetaSabreSwap(
                    coupling_map=coupling_map,
                    max_depth=1
                )
            ),
        translation=pm_lv2.translation,
        optimization=pm_lv2.optimization,
        scheduling=pm_lv2.scheduling
    )
    # print_passes(pm_msabre)

    # pm_msabre_d3 = StagedPassManager(
    # stages=['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling'],
    #     init=pm_lv2.init,
    #     layout=pm_lv2.layout,
    #     routing=PassManager(
    #             MetaSabreSwap(
    #                 coupling_map=coupling_map,
    #                 max_depth=3
    #             )
    #         ),
    #     translation=pm_lv2.translation,
    #     optimization=pm_lv2.optimization,
    #     scheduling=pm_lv2.scheduling
    # )

    # pm_msabre_d10 = StagedPassManager(
    # stages=['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling'],
    #     init=pm_lv2.init,
    #     layout=pm_lv2.layout,
    #     routing=PassManager(
    #             MetaSabreSwap(
    #                 coupling_map=coupling_map,
    #                 max_depth=5
    #             )
    #         ),
    #     translation=pm_lv2.translation,
    #     optimization=pm_lv2.optimization,
    #     scheduling=pm_lv2.scheduling
    # )
    # print_passes(pm_msabre_d10)

    # Grades
    scorer = scorer()
    transpiler_list = [
        pm_lv2,
        pm_msabre,
        # pm_msabre_d3,
        # pm_msabre_d10
    ]
    tr_depths, tr_gate_counts, tr_cnot_counts, tr_scores = grade_transpiler(
        transpiler_list, backend, scorer, num_qubits=np.arange(8, 9)
    )

    for i in range(len(tr_scores)):
        print(f"Score for {i}: {tr_scores[i]}")