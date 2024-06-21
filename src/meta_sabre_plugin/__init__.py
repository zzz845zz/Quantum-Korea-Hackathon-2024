from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager
from meta_sabre.meta_sabre_swap import MetaSabreSwap

# from qiskit.transpiler.passes.routing import SabreSwap as BaseSabreSwap
# from qiskit.transpiler.passes import SabreSwap as BaseSabreSwap


class MetaSabreSwapPlugin(PassManagerStagePlugin):
    def pass_manager(self, pass_manager_config, optimization_level):
        routing_pm = PassManager(
            MetaSabreSwap(
                coupling_map=pass_manager_config.coupling_map,
            )
        )
        # routing_pm = PassManager(
        # BaseSabreSwap(
        #     coupling_map=pass_manager_config.coupling_map,
        # )
        # )
        # layout_pm = PassManager(
        #     [
        #         VF2Layout(
        #             coupling_map=pass_manager_config.coupling_map,
        #             properties=pass_manager_config.backend_properties,
        #             max_trials=optimization_level * 10 + 1,
        #             target=pass_manager_config.target,
        #         )
        #     ]
        # )
        # layout_pm.append(
        #     [
        #         TrivialLayout(pass_manager_config.coupling_map),
        #         SabreLayout(pass_manager_config.coupling_map),
        #     ]
        #     # condition=_vf2_match_not_found,
        # )
        # routing_pm += common.generate_embed_passmanager(
        #     pass_manager_config.coupling_map
        # )
        return routing_pm
