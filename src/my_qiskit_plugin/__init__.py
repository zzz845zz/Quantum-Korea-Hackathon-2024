# # This import is needed for python versions prior to 3.10
# from __future__ import annotations

# from qiskit.transpiler import PassManager
# from qiskit.transpiler.passes import VF2Layout
# from qiskit.transpiler.passmanager_config import PassManagerConfig
# from qiskit.transpiler.preset_passmanagers import common
# from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


# class MyLayoutPlugin(PassManagerStagePlugin):
#     def pass_manager(
#         self,
#         pass_manager_config: PassManagerConfig,
#         optimization_level: int | None = None,
#     ) -> PassManager:
#         layout_pm = PassManager(
#             [
#                 VF2Layout(
#                     coupling_map=pass_manager_config.coupling_map,
#                     properties=pass_manager_config.backend_properties,
#                     max_trials=optimization_level * 10 + 1,
#                     target=pass_manager_config.target,
#                 )
#             ]
#         )
#         layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
#         return layout_pm
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import VF2Layout, TrivialLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
 
def _vf2_match_not_found(property_set):
    return property_set["layout"] is None or (
        property_set["VF2Layout_stop_reason"] is not None
        and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND)
 
 
class VF2LayoutPlugin(PassManagerStagePlugin):
 
    def pass_manager(self, pass_manager_config, optimization_level):
        layout_pm = PassManager(
            [
                VF2Layout(
                    coupling_map=pass_manager_config.coupling_map,
                    properties=pass_manager_config.backend_properties,
                    max_trials=optimization_level * 10 + 1,
                    target=pass_manager_config.target
                )
            ]
        )
        layout_pm.append(
            TrivialLayout(pass_manager_config.coupling_map),
            condition=_vf2_match_not_found,
        )
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm