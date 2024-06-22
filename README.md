
Connect venv (Python 3.12.3)

```bash
source .venv/bin/activate
```

Install dependencies

```bash
(.venv) pip install -r requirements.txt
```

Run

```bash
(.venv) python3 run.py -c qft -n 3 -o qft.csv       # 3 qubits QFT
(.venv) python3 run.py -c random -n 3 -o random.csv # 3 qubits random
```

Bench

```bash
(.venv) ./bench.sh qft
(.venv) ./bench.sh random
```
