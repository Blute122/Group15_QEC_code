"""Group 15 - Quantum Error Correction.
Distance-d repetition codes (d = 3, 5, 7) against bit-flip or phase-flip channels, compared with an
unencoded single-qubit baseline, on Qiskit Aer with the assignment's N0/N1/N2 noise models."""
from itertools import product
from math import comb

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError

SHOTS = 1024
TRIALS = 5                            # repeated trials per configuration


def seed_for(*key):
    """Independent simulator seed for (configuration, trial).
    Aer seeds shot k with (seed + k); seeds closer than the shot count would share most of their
    random stream, so seeds are drawn from a NumPy SeedSequence keyed on the configuration instead."""
    return int(np.random.default_rng([int(k) for k in key]).integers(2**31 - 1))   # integer keys only (reproducible)
CHANNEL_P = [0.001, 0.01, 0.02]       # assignment noise levels: 0.1 %, 1 %, 2 %

# Common-settings noise models (depolarizing + readout), as specified for all groups
NOISE_LEVELS = {
    "N0": None,
    "N1": dict(p1=0.001, p2=0.005, ro=0.01),
    "N2": dict(p1=0.01, p2=0.02, ro=0.03),
}


# ---------------------------------------------------------------- noise
def build_noise_model(level):
    """Return an Aer NoiseModel for N1/N2 (None for the ideal N0 case)."""
    cfg = NOISE_LEVELS[level]
    if cfg is None:
        return None
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(cfg["p1"], 1), ["h", "x", "z"])
    nm.add_all_qubit_quantum_error(depolarizing_error(cfg["p2"], 2), ["cx"])
    ro = cfg["ro"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def channel(kind, p):
    """The 'noisy channel' under study: X (bit flip) or Z (phase flip) with probability p."""
    return pauli_error([("X" if kind == "bit" else "Z", p), ("I", 1 - p)])


def inject(qc, q, kind, p=0.0, force=False):
    if force:
        (qc.x if kind == "bit" else qc.z)(q)
    elif p > 0:
        qc.append(channel(kind, p), [q])


# ---------------------------------------------------------------- circuits
def unencoded(kind, p=0.0, force=False):
    """Baseline: one physical qubit. Bit test stores |0>, phase test stores |+> (measured in X basis)."""
    qc = QuantumCircuit(1, 1, name=f"unencoded_{kind}")
    if kind == "phase":
        qc.h(0)
    qc.barrier(label="channel")
    inject(qc, 0, kind, p, force)
    qc.barrier()
    if kind == "phase":
        qc.h(0)
    qc.measure(0, 0)
    return qc


def decoder_table(d):
    """Minimum-weight decoder for a distance-d repetition code.
    Syndrome bit j = parity(d_j, d_{j+1}). Maps each nonzero syndrome integer to the data qubits to flip."""
    table = {}
    for bits in product([0, 1], repeat=d - 1):
        s = sum(b << j for j, b in enumerate(bits))
        if s == 0:
            continue
        e = [0]
        for b in bits:
            e.append(e[-1] ^ b)
        if sum(e) > d // 2:                      # pick the lower-weight of the two consistent patterns
            e = [1 - x for x in e]
        table[s] = [i for i, x in enumerate(e) if x]
    return table


def repetition_code(d, kind, p=0.0, force_on=(), readout="decode"):
    """Distance-d repetition code: d data qubits + (d-1) ancillas, one round of
    syndrome extraction, real-time (feed-forward) correction, then logical readout.
    readout="decode":   unitary decoding (reverse fan-out CNOTs) and measurement of d0 only.
    readout="majority": every data qubit is measured and the logical bit is their majority vote."""
    data = QuantumRegister(d, "d")
    anc = QuantumRegister(d - 1, "anc")
    syn = ClassicalRegister(d - 1, "syn")
    out = ClassicalRegister(1 if readout == "decode" else d, "out")
    qc = QuantumCircuit(data, anc, syn, out, name=f"rep{d}_{kind}")

    for i in range(1, d):                        # encode |x> -> |x...x>
        qc.cx(data[0], data[i])
    if kind == "phase":                          # |0> -> |+...+>, |1> -> |-...->
        qc.h(data)
    qc.barrier(label="channel")
    for i in range(d):
        inject(qc, data[i], kind, p, force=(i in force_on))
    qc.barrier()
    if kind == "phase":                          # rotate Z errors into X errors
        qc.h(data)
    for j in range(d - 1):                       # parity checks Z_j Z_{j+1}
        qc.cx(data[j], anc[j])
        qc.cx(data[j + 1], anc[j])
    qc.measure(anc, syn)
    for s, flips in decoder_table(d).items():    # conditional correction
        with qc.if_test((syn, s)):
            for i in flips:
                qc.x(data[i])
    if readout == "decode":
        for i in reversed(range(1, d)):          # decode
            qc.cx(data[0], data[i])
        qc.measure(data[0], out[0])
    else:
        qc.measure(data, out)
    return qc


def build(code, kind, p=0.0, force_on=(), readout="decode"):
    if code == 1:
        return unencoded(kind, p, force=bool(force_on))
    return repetition_code(code, kind, p, force_on, readout)


# ---------------------------------------------------------------- running and metrics
def run_counts(qc, noise_level="N0", seed=11, shots=SHOTS):
    sim = AerSimulator(noise_model=build_noise_model(noise_level), seed_simulator=seed)
    return sim.run(qc, shots=shots).result().get_counts()


def metrics(counts):
    """Logical error rate; and for codes, recovery success = P(correct output | nonzero syndrome)."""
    total = sum(counts.values())
    wrong = det = det_ok = 0
    for key, c in counts.items():
        parts = key.split()
        o = parts[0]
        logical = o if len(o) == 1 else str(int(o.count("1") > len(o) // 2))   # majority vote
        if logical != "0":
            wrong += c
        if len(parts) > 1 and "1" in parts[1]:
            det += c
            if logical == "0":
                det_ok += c
    return dict(ler=wrong / total, detected=det, detected_ok=det_ok, shots=total)


def theory_ler(d, p):
    """Code-capacity logical error: more than half of the d qubits flipped (d = 1 is unencoded)."""
    return sum(comb(d, k) * p**k * (1 - p) ** (d - k) for k in range(d // 2 + 1, d + 1))


def resources(code, kind, readout="decode"):
    """Resource counts of the error-correction circuit itself (channel removed)."""
    qc = build(code, kind, p=0.0, readout=readout)
    raw = qc.count_ops()
    cond_x = sum(len(inst.operation.blocks[0].data) for inst in qc.data if inst.operation.name == "if_else")
    # transpile to an IBM-style native basis (CX, RZ, SX, X), all-to-all connectivity, no optimisation
    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=0, seed_transpiler=11)
    ops = tq.count_ops()
    native_1q = ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0)
    return {
        "physical qubits": qc.num_qubits,
        "data / ancilla": f"{code} / {qc.num_qubits - code}",
        "two-qubit gates (CX)": ops.get("cx", 0),
        "single-qubit gates (H)": raw.get("h", 0),
        "conditional X gates (decoder branches)": f"{cond_x} ({raw.get('if_else', 0)})",
        "native 1-qubit gates (RZ/SX/X)": native_1q,
        "total gates (CX + native 1q + conditional X)": ops.get("cx", 0) + native_1q + cond_x,
        "measurements": ops.get("measure", 0),
        "transpiled depth": tq.depth(lambda i: i.operation.name != "barrier"),
    }
