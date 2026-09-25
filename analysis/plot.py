"""Turn DOK results into blog charts (PNG).

  python analysis/plot.py --results results --out blog/images
Charts whose input file is missing are skipped; the memory-wall chart needs no data.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots (fixed order) + neutral inks
SERIES = {"qpp-cpu": "#2a78d6", "nvidia": "#eb6834", "nvidia-fp64": "#1baf7a", "tensornet": "#4a3aa7"}
LABELS = {"qpp-cpu": "CPU (qpp-cpu)", "nvidia": "GPU fp32 (nvidia)",
          "nvidia-fp64": "GPU fp64 (nvidia fp64)", "tensornet": "GPU tensor network"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def memory_wall(out):
    """State-vector memory 2^n x bytes vs qubits, with real device capacities."""
    ns = list(range(24, 41))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ns, [2**n * 8 / 2**30 for n in ns], color=SERIES["nvidia"], label="complex64 (fp32)")
    ax.plot(ns, [2**n * 16 / 2**30 for n in ns], color=SERIES["nvidia-fp64"], label="complex128 (fp64)")
    for gib, name in [(16, "Laptop RAM 16 GB"), (32, "V100 32 GB"), (80, "H100 80 GB"),
                      (640, "H100 x8 = 640 GB")]:
        ax.axhline(gib, color=INK2, linewidth=0.8, linestyle="--")
        ax.text(24.2, gib * 1.08, name, color=INK2, fontsize=9, va="bottom")
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.1, 2**14)
    ticks = [0.125, 1, 8, 64, 512, 4096]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["128 MB", "1 GB", "8 GB", "64 GB", "512 GB", "4 TB"])
    ax.set_xlabel("qubits n")
    ax.set_ylabel("state vector memory")
    ax.set_title("Every extra qubit doubles the memory", loc="left", fontsize=12)
    ax.legend(loc="lower right")
    save(fig, out, "memory_wall.png")


def scaling(results, out):
    path = os.path.join(results, "scaling.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:  # empty when that run crashed
        return
    df = pd.read_csv(path)
    ok = df[df.status == "ok"]
    circuits = [c for c in ["ghz", "qft", "random"] if c in set(ok.circuit)]
    fig, axes = plt.subplots(1, len(circuits), figsize=(4.2 * len(circuits), 4), sharey=True)
    axes = [axes] if len(circuits) == 1 else axes
    for ax, c in zip(axes, circuits):
        for t in SERIES:
            d = ok[(ok.circuit == c) & (ok.target == t)].sort_values("qubits")
            if len(d):
                ax.plot(d.qubits, d.seconds, marker="o", color=SERIES[t], label=LABELS[t])
        ax.set_yscale("log")
        ax.set_title(c.upper(), loc="left", fontsize=12)
        ax.set_xlabel("qubits n")
    axes[0].set_ylabel("wall time (s, log)")
    axes[-1].legend(loc="upper left")
    save(fig, out, "scaling_time.png")

    # speedup = CPU time / GPU time at the same n
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for c, style in zip(circuits, ["-", "--", ":"]):
        cpu = ok[(ok.circuit == c) & (ok.target == "qpp-cpu")].set_index("qubits").seconds
        gpu = ok[(ok.circuit == c) & (ok.target == "nvidia")].set_index("qubits").seconds
        s = (cpu / gpu).dropna()
        if len(s):
            ax.plot(s.index, s.values, style, marker="o", color=SERIES["nvidia"], label=c.upper())
            ax.annotate(f"{s.values[-1]:.0f}x", (s.index[-1], s.values[-1]),
                        textcoords="offset points", xytext=(6, 0), color=INK, fontsize=10)
    ax.axhline(1, color=INK2, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("qubits n")
    ax.set_ylabel("speedup: CPU time / GPU fp32 time")
    ax.set_title("GPU speedup over CPU", loc="left", fontsize=12)
    ax.legend(loc="upper left")
    save(fig, out, "speedup.png")

    # largest n reached per target
    mx = ok.groupby("target").qubits.max().reindex([t for t in SERIES if t in set(ok.target)])
    mx.to_csv(os.path.join(out, "max_qubits.csv"))


def tensornet(results, out):
    path = os.path.join(results, "tensornet.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:  # empty when that run crashed
        return
    df = pd.read_csv(path)
    ok = df[df.status == "ok"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for c, style in [("ghz", "-"), ("random", "--")]:
        d = ok[ok.circuit == c].sort_values("qubits")
        if len(d):
            ax.plot(d.qubits, d.seconds, style, marker="o", color=SERIES["tensornet"], label=c.upper())
    ax.set_xlabel("qubits n")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Tensor networks: beyond the state-vector wall (shallow circuits)", loc="left", fontsize=12)
    ax.legend(loc="upper left")
    save(fig, out, "tensornet.png")


def vqe(results, out):
    path = os.path.join(results, "vqe.json")
    if not os.path.exists(path):
        return
    data = json.load(open(path))
    mols = data["molecules"]
    if mols:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        names = [f"{m['molecule']}\n{m['qubits']} qubits" for m in mols]
        errs = [max(m["error"], 1e-10) for m in mols]
        ax.bar(names, errs, color=SERIES["nvidia-fp64"], width=0.5)
        for i, e in enumerate(errs):
            ax.text(i, e * 1.3, f"{e:.1e}", ha="center", color=INK, fontsize=9)
        ax.axhline(1.6e-3, color=INK2, linestyle="--", linewidth=1)
        ax.text(len(errs) - 0.5, 1.6e-3 * 1.3, "chemical accuracy 1.6 mHa", ha="right",
                color=INK2, fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("|E_VQE - E_exact| (Hartree)")
        ax.set_title("VQE error vs exact answer (lower is better)", loc="left", fontsize=12)
        ax.grid(axis="x", visible=False)
        save(fig, out, "vqe_error.png")
    curve = data.get("h2_curve", [])
    if curve:
        c = pd.DataFrame(curve)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(c.distance, c.hf_energy, color=SERIES["qpp-cpu"], label="Hartree-Fock (classical approx.)")
        ax.plot(c.distance, c.exact_energy, color=INK2, linestyle="--", label="FCI (exact)")
        ax.plot(c.distance, c.vqe_energy, "o", color=SERIES["nvidia"], markersize=7, label="VQE (simulated quantum)")
        ax.set_xlabel("H-H distance (Angstrom)")
        ax.set_ylabel("energy (Hartree)")
        ax.set_title("H2 dissociation curve", loc="left", fontsize=12)
        ax.legend(loc="upper right")
        save(fig, out, "h2_curve.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="blog/images")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    memory_wall(a.out)
    scaling(a.results, a.out)
    tensornet(a.results, a.out)
    vqe(a.results, a.out)


if __name__ == "__main__":
    main()
