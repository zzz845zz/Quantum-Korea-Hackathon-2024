from dataclasses import dataclass
from json import dump, load

from qiskit_ibm_runtime import QiskitRuntimeService
from qkh2024.grader import scorer
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.passes.synthesis.plugin import unitary_synthesis_plugin_names
from qiskit.circuit.library import QFT
from qiskit import transpile
from tqdm import tqdm


layout_method_list = list_stage_plugins("layout") + [""]
routing_method_list = list_stage_plugins("routing") + [""]
translation_method_list = list_stage_plugins("translation") + [""]
scheduling_method_list = list_stage_plugins("scheduling") + [""]
unitary_synthesis_method_list = unitary_synthesis_plugin_names()
init_method_list = list_stage_plugins("init") + [""]
optimization_method_list = list_stage_plugins("optimization") + [""]


@dataclass
class Mutant:
    layout_method: str
    routing_method: str
    translation_method: str
    scheduling_method: str
    # approximation_degree: float  # 1.0: exact, 0.0: approximate
    unitary_synthesis_method: str
    init_method: str
    optimization_method: str

    def __str__(self):
        return ",".join(
            [
                self.layout_method,
                self.routing_method,
                self.translation_method,
                self.scheduling_method,
                self.unitary_synthesis_method,
                self.init_method,
                self.optimization_method,
            ]
        )

    @staticmethod
    def from_str(s: str):
        return Mutant(*s.split(","))


mutant_list = [
    Mutant(l, r, t, s, u, i, o)
    for l in layout_method_list
    for r in routing_method_list
    for t in translation_method_list
    for s in scheduling_method_list
    for u in unitary_synthesis_method_list
    for i in init_method_list
    for o in optimization_method_list
    if l != "none"
    and r not in {"none", "lookahead"}
    and t != "none"
    and s != "none"
    and u != "none"
    and i != "none"
    and o != "none"
]


def brute_force(nq):
    json_file_name = f"mut_scores_{nq}.json"
    try:
        mutant_dict = load(open(json_file_name))
    except:
        print("No file found")
        mutant_dict = {}

    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
    )
    backend = service.backend("ibm_sherbrooke")

    circuit = QFT(nq)
    new_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=3,
    )

    base_score = scorer().score(new_circuit, backend)

    for mutant in tqdm(mutant_list):
        # print(mutant)
        # if i in {35}:
        #     continue
        if str(mutant) in mutant_dict:
            continue

        circuit = QFT(nq)
        new_circuit = transpile(
            circuit,
            backend=backend,
            layout_method=mutant.layout_method or None,
            routing_method=mutant.routing_method or None,
            translation_method=mutant.translation_method or None,
            scheduling_method=mutant.scheduling_method or None,
            unitary_synthesis_method=mutant.unitary_synthesis_method or None,
            init_method=mutant.init_method or None,
            optimization_method=mutant.optimization_method or None,
            optimization_level=3,
        )
        score = scorer().score(new_circuit, backend)
        mutant_dict[str(mutant)] = score

        if score > base_score:
            print(f"{mutant}: {score} > {base_score}")

        with open(json_file_name, "w") as f:
            dump(mutant_dict, f)

    leaderboards = sorted(mutant_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"default: {base_score}")

    for i, (k, v) in enumerate(leaderboards[:10]):
        print(f"#{i+1}: {k} - {v}")


if __name__ == "__main__":
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="fa0372ac79105aaec3e2bbff758cb43dc9506244ea5fba95957381cd14f56a38fc96f0fbc31e98d93318017772027e45f80ab6e71678e18d44e05f8f6655516b",
    )
    backend = service.backend("ibm_sherbrooke")

    for nq in range(3, 15):
        bf_dict = load(open(f"mut_scores_{nq}.json"))
        leaderboards = sorted(bf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        best_mutants = [Mutant.from_str(l[0]) for l in leaderboards]
        circuit = QFT(nq)
        base_circuit = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
        )
        best_mutant = None
        best_score = 0
        for m in best_mutants:
            best_circuit = transpile(
                circuit,
                backend=backend,
                layout_method=m.layout_method or None,
                routing_method=m.routing_method or None,
                translation_method=m.translation_method or None,
                scheduling_method=m.scheduling_method or None,
                unitary_synthesis_method=m.unitary_synthesis_method or None,
                init_method=m.init_method or None,
                optimization_method=m.optimization_method or None,
                optimization_level=3,
            )
            if scorer().score(best_circuit, backend) > best_score:
                best_mutant = m
                best_score = scorer().score(best_circuit, backend)

        base_score = scorer().score(base_circuit, backend)

        print(f"{nq:2d} qubits: {base_score:5f} (base) -> {best_score:5f} ({m})")
