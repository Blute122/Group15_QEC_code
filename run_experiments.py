"""Runs every experiment for the Group 15 report and writes CSV files to ./results.
Usage:  python run_experiments.py"""
import itertools, os, platform, time
import numpy as np, pandas as pd
import qiskit, qiskit_aer
from qec import (SHOTS, TRIALS, seed_for, CHANNEL_P, NOISE_LEVELS, build, run_counts, metrics, theory_ler, resources)

os.makedirs("results", exist_ok=True)
t0 = time.time()

# ---- 1. controlled-error verification (deterministic injection, every error pattern) ----
rows = []
for d, kind in itertools.product([1, 3, 5, 7], ["bit", "phase"]):
    n = 1 if d == 1 else d
    for pat in itertools.product([0, 1], repeat=n):
        f = tuple(i for i, x in enumerate(pat) if x)
        m = metrics(run_counts(build(d, kind, force_on=f), seed=seed_for(0, d), shots=SHOTS))
        rows.append(dict(code=d, kind=kind, weight=len(f), pattern=str(f), ler=m["ler"]))
ver = pd.DataFrame(rows)
ver.to_csv("results/verification_all_patterns.csv", index=False)
ver.groupby(["code", "kind", "weight"])["ler"].agg(["count", "min", "max"]).reset_index() \
   .to_csv("results/verification_summary.csv", index=False)

# ---- 2. main grid: code x readout x noise model x channel p x 5 seeds ----
configs = [(1, "none")] + [(d, ro) for d in (3, 5, 7) for ro in ("decode", "majority")]
rows = []
for kind, (d, ro), lvl, p in itertools.product(["bit", "phase"], configs, NOISE_LEVELS, CHANNEL_P):
    qc = build(d, kind, p, readout=ro if d > 1 else "decode")
    for t in range(TRIALS):
        seed = seed_for(1, ["bit", "phase"].index(kind), d, ["none", "decode", "majority"].index(ro),
                        int(lvl[1]), int(round(p * 1000)), t)
        m = metrics(run_counts(qc, lvl, seed=seed))
        rows.append(dict(kind=kind, code=d, readout=ro, noise=lvl, p=p, trial=t, seed=seed, **m))
grid = pd.DataFrame(rows)
grid.to_csv("results/grid_raw.csv", index=False)

agg = grid.groupby(["kind", "code", "readout", "noise", "p"]).agg(
    ler_mean=("ler", "mean"), ler_sd=("ler", lambda x: x.std(ddof=1)),
    detected=("detected", "sum"), detected_ok=("detected_ok", "sum"), shots=("shots", "sum")).reset_index()
agg["recovery_success"] = np.where(agg["detected"] > 0, agg["detected_ok"] / agg["detected"].clip(lower=1), np.nan)
agg["theory_N0"] = [theory_ler(d, p) for d, p in zip(agg["code"], agg["p"])]
agg.to_csv("results/grid_summary.csv", index=False)

# ---- 3. supplementary sweep (ideal gates): where does each code stop helping? ----
P_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
rows = []
for kind, d, p in itertools.product(["bit", "phase"], [1, 3, 5, 7], P_SWEEP):
    qc = build(d, kind, p)
    for t in range(TRIALS):
        seed = seed_for(2, ["bit", "phase"].index(kind), d, int(round(p * 1000)), t)
        rows.append(dict(kind=kind, code=d, p=p, trial=t, seed=seed, ler=metrics(run_counts(qc, "N0", seed=seed))["ler"]))
sw = pd.DataFrame(rows)
sw = sw.groupby(["kind", "code", "p"]).agg(ler_mean=("ler", "mean"), ler_sd=("ler", lambda x: x.std(ddof=1))).reset_index()
sw["theory"] = [theory_ler(d, p) for d, p in zip(sw["code"], sw["p"])]
sw.to_csv("results/sweep_N0.csv", index=False)

# ---- 4. syndrome histograms (one seed) ----
hist = {}
for lvl in ["N0", "N2"]:
    hist[lvl] = run_counts(build(3, "bit", 0.02), lvl, seed=seed_for(3, int(lvl[1])))
pd.DataFrame([dict(noise=l, key=k, count=c) for l, cs in hist.items() for k, c in cs.items()]) \
  .to_csv("results/histogram_rep3_bit_p2.csv", index=False)

# ---- 5. resources ----
res = {f"{'Unencoded' if d == 1 else f'Rep-{d}'} ({kind}{'' if d == 1 else ', ' + ro})": resources(d, kind, ro if d > 1 else "decode")
       for d, kind, ro in [(1, "bit", "-"), (1, "phase", "-"), (3, "bit", "decode"), (3, "phase", "decode"),
                           (3, "bit", "majority"), (5, "bit", "decode"), (5, "bit", "majority"),
                           (7, "bit", "decode"), (7, "bit", "majority"), (7, "phase", "decode")]}
pd.DataFrame(res).to_csv("results/resources.csv")

with open("results/environment.txt", "w") as f:
    f.write(f"python {platform.python_version()}\nqiskit {qiskit.__version__}\nqiskit-aer {qiskit_aer.__version__}\n"
            f"numpy {np.__version__}\npandas {pd.__version__}\nplatform {platform.platform()}\n"
            f"cpu_count {os.cpu_count()}\nruntime_s {time.time() - t0:.1f}\n")
print(f"done in {time.time() - t0:.1f}s")
