# Quantum-Korea-Hackathon-2024

### Topic: Quantum Circuit Transpilation

### Background:
Transpilation of a quantum circuit is a complex process of converting the given circuit into the executable set of gates, matching with the topology of the target quantum device, using an optimal number of gates, and applying techniques to deal with errors. Depending on the performance of the transpiler, a circuit can shrink into one-fifth of the original circuit and achieve a significant reduction of errors physically. In this topic, you will explore the detailed steps of transpilation and play with ready-made or your own functions to optimize the given quantum circuit. The target circuit of this topic will be the Quantum Fourier Transform (QFT) circuit, which is one of the most important quantum subroutines, providing exponential speedup compared to classical computation.

### Problem:
Transpile the QFT circuit to get the best score. The circuit must perform the QFT with arbitrary inputs, matching with the target quantum devcie topology within the basic gate set. The score will be given based on the expected performance of the circuit, considering accuracy of each gate in the circuit.

### Reference:
- Kremer et al., Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning https://arxiv.org/abs/2405.13196 (2024).
- Bäumer et al., Quantum Fourier Transform using Dynamic Circuits, https://arxiv.org/abs/2403.09514 (2024).

### Additional Reference (IBM resources):
- Introduction to transpilation: https://docs.quantum.ibm.com/transpile
- AI Transpiler service: https://docs.quantum.ibm.com/transpile/qiskit-transpiler-service
- Transpiler plugins: https://docs.quantum.ibm.com/api/qiskit/transpiler_plugins#module-qiskit.transpiler.preset_passmanagers.plugin
- Qiskit ecosystem: https://qiskit.github.io/ecosystem/
