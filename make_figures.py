"""Builds every figure in ./figures from ./results (run run_experiments.py first)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from qec import build, theory_ler, decoder_table

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight"})
INK, MUTED = "#0b0b0b", "#52514e"
CODE_STYLE = {1: dict(color=MUTED, marker="o", ls="--", label="Unencoded (1 qubit)"),
              3: dict(color="#2a78d6", marker="s", ls="-", label="Rep-3 (5 qubits)"),
              5: dict(color="#eb6834", marker="^", ls="-", label="Rep-5 (9 qubits)"),
              7: dict(color="#1baf7a", marker="D", ls="-", label="Rep-7 (13 qubits)")}
g = pd.read_csv("results/grid_summary.csv")
sw = pd.read_csv("results/sweep_N0.csv")


def sel(kind, code, readout, noise):
    ro = "none" if code == 1 else readout
    return g[(g.kind == kind) & (g.code == code) & (g.readout == ro) & (g.noise == noise)].sort_values("p")


# ---- Fig 1-4: circuits ----
fig, axes = plt.subplots(1, 2, figsize=(9, 1.9))
for ax, kind in zip(axes, ["bit", "phase"]):
    build(1, kind, 0.01).draw("mpl", ax=ax, fold=-1)
    ax.set_title(f"Unencoded baseline – {kind}-flip test", fontsize=10)
fig.savefig("figures/fig1_unencoded.png"); plt.close(fig)

for kind, name in [("bit", "fig2_rep3_bit.png"), ("phase", "fig3_rep3_phase.png")]:
    f = build(3, kind, 0.01).draw("mpl", fold=-1, scale=0.7)
    f.savefig(f"figures/{name}"); plt.close(f)

# Largest configuration: the 63-branch lookup is drawn as one labelled block so the diagram stays legible
d = 7
data, anc = QuantumRegister(d, "d"), QuantumRegister(d - 1, "anc")
syn, out = ClassicalRegister(d - 1, "syn"), ClassicalRegister(1, "out")
qc = QuantumCircuit(data, anc, syn, out)
for i in range(1, d):
    qc.cx(data[0], data[i])
qc.barrier(label="channel")
for i in range(d):
    qc.append(Instruction("X/Z ch.", 1, 0, []), [data[i]])
qc.barrier()
for j in range(d - 1):
    qc.cx(data[j], anc[j]); qc.cx(data[j + 1], anc[j])
qc.measure(anc, syn)
qc.append(Instruction(f"lookup correction ({len(decoder_table(d))} branches)", d, d - 1, []), list(data), list(syn))
for i in reversed(range(1, d)):
    qc.cx(data[0], data[i])
qc.measure(data[0], out[0])
f = qc.draw("mpl", fold=-1, scale=0.6)
f.savefig("figures/fig4_rep7_bit.png"); plt.close(f)

# ---- Fig 5: logical error rate vs channel error, per noise model ----
for kind in ["bit", "phase"]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, lvl in zip(axes, ["N0", "N1", "N2"]):
        for code in [1, 3, 5, 7]:
            s = sel(kind, code, "decode", lvl)
            st = CODE_STYLE[code]
            ax.errorbar(s.p * 100, s.ler_mean, yerr=s.ler_sd, capsize=3, lw=2, ms=5, **st)
        ax.set_title({"N0": "N0 – ideal gates", "N1": "N1 – low gate/readout noise",
                      "N2": "N2 – moderate gate/readout noise"}[lvl], fontsize=10)
        ax.set_xlabel("channel error probability p (%)")
        ax.set_xticks([0.1, 1, 2])
    axes[0].set_ylabel("logical error rate (mean ± SD, 5 seeds)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{kind.capitalize()}-flip channel: logical error rate vs channel noise (1024 shots × 5 seeds)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"figures/fig5_ler_vs_p_{kind}.png"); plt.close(fig)

# ---- Fig 6: readout strategy ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
w = 0.36
for ax, lvl in zip(axes, ["N0", "N1", "N2"]):
    x = np.arange(3)
    for k, (ro, col, lab) in enumerate([("decode", "#2a78d6", "unitary decode, read 1 qubit"),
                                        ("majority", "#eb6834", "measure all data, majority vote")]):
        vals = [sel("bit", d, ro, lvl).query("p == 0.02") for d in (3, 5, 7)]
        ax.bar(x + (k - 0.5) * w, [v.ler_mean.iloc[0] for v in vals], w * 0.92, yerr=[v.ler_sd.iloc[0] for v in vals],
               color=col, capsize=3, label=lab)
    base = sel("bit", 1, "none", lvl).query("p == 0.02")
    ax.axhline(base.ler_mean.iloc[0], color=MUTED, ls="--", lw=1.5, label="unencoded baseline")
    ax.set_xticks(x, ["Rep-3", "Rep-5", "Rep-7"])
    ax.set_title(f"{lvl}, p = 2 %", fontsize=10)
axes[0].set_ylabel("logical error rate (mean ± SD)")
hd, lb = axes[0].get_legend_handles_labels()
fig.suptitle("Effect of the logical read-out strategy (bit-flip channel, p = 2 %)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(hd, lb, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.savefig("figures/fig6_readout_strategy.png"); plt.close(fig)

# ---- Fig 7: supplementary sweep under ideal gates ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
pg = np.linspace(0.005, 0.6, 200)
for code in [1, 3, 5, 7]:
    s = sw[(sw.kind == "bit") & (sw.code == code)]
    st = CODE_STYLE[code]
    for ax in axes:
        ax.plot(pg, [theory_ler(code, p) for p in pg], color=st["color"], lw=1, alpha=0.6)
        ax.errorbar(s.p, s.ler_mean, yerr=s.ler_sd, fmt=st["marker"], color=st["color"], ms=5, capsize=2, label=st["label"])
axes[0].axvline(0.5, color=MUTED, lw=0.8); axes[0].text(0.505, 0.05, "break-even\np = 0.5", fontsize=8, color=MUTED)
axes[0].set_xlabel("channel error probability p"); axes[0].set_ylabel("logical error rate")
axes[0].set_title("Linear scale (lines: binomial theory)", fontsize=10)
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_ylim(1e-4, 1)
axes[1].set_xlabel("channel error probability p"); axes[1].set_title("Log–log scale (zeros not shown)", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Supplementary sweep – bit-flip channel, ideal gates (N0), 1024 shots × 5 seeds", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig7_sweep_N0.png"); plt.close(fig)

# ---- Fig 8: syndrome histograms ----
h = pd.read_csv("results/histogram_rep3_bit_p2.csv", dtype={"key": str})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
order = [f"{o} {s}" for o in "01" for s in ["00", "01", "11", "10"]]
for ax, lvl in zip(axes, ["N0", "N2"]):
    hh = h[h.noise == lvl].set_index("key")["count"].reindex(order).fillna(0)
    cols = ["#2a78d6" if k.startswith("0") else "#eb6834" for k in order]
    ax.bar(range(len(order)), hh.values, color=cols)
    for i, v in enumerate(hh.values):
        ax.text(i, v + 8, int(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(order)), [k.replace(" ", "\n") for k in order], fontsize=8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title(f"Rep-3 bit-flip code, p = 2 %, {lvl}", fontsize=10)
    ax.set_xlabel("logical output (top) / syndrome s1s0 (bottom)")
axes[0].set_ylabel("counts (of 1024, symlog)")
fig.suptitle("Measurement-count histograms (blue = correct logical output, orange = logical error)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig8_histograms.png"); plt.close(fig)

# ---- Fig 9: error vs resource cost ----
res = pd.read_csv("results/resources.csv", index_col=0)
cx = {"decode": {1: 0, 3: 8, 5: 16, 7: 24}, "majority": {1: 0, 3: 6, 5: 12, 7: 18}}
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for lvl, col in [("N0", "#2a78d6"), ("N1", "#eb6834"), ("N2", "#1baf7a")]:
    for ro, ls, mk in [("decode", "-", "o"), ("majority", ":", "s")]:
        xs, ys, es = [], [], []
        for code in [1, 3, 5, 7]:
            s = sel("bit", code, ro, lvl).query("p == 0.02")
            xs.append(cx[ro][code]); ys.append(s.ler_mean.iloc[0]); es.append(s.ler_sd.iloc[0])
        ax.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker=mk, ms=5, capsize=3, label=f"{lvl}, {ro}")
ax.set_xticks([0, 6, 8, 12, 16, 18, 24])
ax.set_xlabel("two-qubit (CX) gates per logical qubit")
ax.set_ylabel("logical error rate at p = 2 %")
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_title("Error reduction vs resource cost (bit-flip channel)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig9_cost_vs_benefit.png"); plt.close(fig)
print("figures written:", sorted(os.listdir("figures")))
