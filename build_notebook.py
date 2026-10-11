import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())
nb.cells = [
    md("""
# Group 15 – Quantum Error Correction
Repetition codes (3, 5, 7 data qubits) vs an unencoded baseline under bit-flip / phase-flip channels
(p = 0.1 %, 1 %, 2 %) and the N0 / N1 / N2 noise models. 1024 shots × 5 independent seeds per configuration.

Files: `qec.py` (circuits, noise models, metrics) · `run_experiments.py` (all runs → `results/`) ·
`make_figures.py` (all plots → `figures/`).
"""),
    code("""
import platform, numpy as np, pandas as pd, qiskit, qiskit_aer
print("Python     ", platform.python_version())
print("Qiskit     ", qiskit.__version__)
print("Qiskit Aer ", qiskit_aer.__version__)
print("NumPy      ", np.__version__, "| pandas", pd.__version__)
from qec import *
print("Shots per run:", SHOTS, "| trials:", TRIALS, "| channel p:", CHANNEL_P)
print("Noise models:", NOISE_LEVELS)
"""),
    md("## Noise model (depolarizing + readout) attached to AerSimulator"),
    code("""
import inspect
print(inspect.getsource(build_noise_model))
print(inspect.getsource(run_counts))
"""),
    md("## Rep-3 bit-flip code circuit"),
    code("""build(3, "bit", 0.02).draw("mpl", fold=-1, scale=0.7)"""),
    md("## A successful run: Rep-3 vs unencoded, bit-flip channel p = 2 %, N1 noise, 5 seeds"),
    code("""
rows = []
for name, d, ro in [("Unencoded", 1, "decode"), ("Rep-3 decode", 3, "decode"), ("Rep-3 majority", 3, "majority")]:
    lers = [metrics(run_counts(build(d, "bit", 0.02, readout=ro), "N1", seed=seed_for(9, d, len(ro), t)))["ler"]
            for t in range(TRIALS)]
    rows.append(dict(config=name, ler_mean=np.mean(lers), ler_sd=np.std(lers, ddof=1), per_seed=np.round(lers, 4).tolist()))
pd.DataFrame(rows)
"""),
    md("## Sample output: one run's measurement counts ('logical syndrome')"),
    code("""
counts = run_counts(build(3, "bit", 0.02), "N0", seed=seed_for(9, 0))
print(dict(sorted(counts.items(), key=lambda kv: -kv[1])))
print(metrics(counts))
"""),
    md("## Summary of the full experiment grid (from results/grid_summary.csv)"),
    code("""
g = pd.read_csv("results/grid_summary.csv")
t = g[(g.kind == "bit") & (g.p == 0.02)].copy()
t["LER (mean ± SD)"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(t.ler_mean, t.ler_sd)]
t.pivot_table(index=["code", "readout"], columns="noise", values="LER (mean ± SD)", aggfunc="first")
"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Group15_QEC.ipynb")
