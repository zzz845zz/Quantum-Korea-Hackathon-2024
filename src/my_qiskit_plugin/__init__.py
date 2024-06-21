from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import VF2Layout, TrivialLayout, SabreLayout


class VF2LayoutPlugin(PassManagerStagePlugin):

    def pass_manager(self, pass_manager_config, optimization_level):
        layout_pm = PassManager(
            [
                VF2Layout(
                    coupling_map=pass_manager_config.coupling_map,
                    properties=pass_manager_config.backend_properties,
                    max_trials=optimization_level * 10 + 1,
                    target=pass_manager_config.target,
                )
            ]
        )
        layout_pm.append(
            [
                TrivialLayout(pass_manager_config.coupling_map),
                SabreLayout(pass_manager_config.coupling_map),
            ]
            # condition=_vf2_match_not_found,
        )
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm
