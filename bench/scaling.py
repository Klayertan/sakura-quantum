"""CPU vs GPU qubit-scaling benchmark for CUDA-Q simulators.

Runs GHZ, QFT and random-layer circuits at increasing qubit counts on each
target and appends one CSV row per run, so partial results survive a crash
or an out-of-memory error.
"""
import argparse
import csv
import math
import os
import random
import subprocess
import time

import cudaq


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


@cudaq.kernel
def qft(n: int, angles: list[float]):
    # angles[k] = pi / 2**k, precomputed on the host
    q = cudaq.qvector(n)
    x(q[0])
    x(q[n - 1])
    for i in range(n):
        h(q[i])
        for j in range(i + 1, n):
            r1.ctrl(angles[j - i], q[j], q[i])
    mz(q)


@cudaq.kernel
def random_layers(n: int, layers: int, thetas: list[float]):
    q = cudaq.qvector(n)
    for l in range(layers):
        for i in range(n):
            ry(thetas[(l * n + i) * 2], q[i])
            rz(thetas[(l * n + i) * 2 + 1], q[i])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
    mz(q)


# name -> (cudaq target, target option, bytes per amplitude)
TARGETS = {
    "qpp-cpu": ("qpp-cpu", None, 16),
    "nvidia": ("nvidia", "fp32", 8),
    "nvidia-fp64": ("nvidia", "fp64", 16),
    "tensornet": ("tensornet", None, None),
}


def set_target(name):
    target, option, _ = TARGETS[name]
    if option:
        cudaq.set_target(target, option=option)
    else:
        cudaq.set_target(target)


def run_circuit(circuit, n, shots, layers, seed):
    if circuit == "ghz":
        counts = cudaq.sample(ghz, n, shots_count=shots)
        ok = {k for k, _ in counts.items()} <= {"0" * n, "1" * n}
    elif circuit == "qft":
        angles = [math.pi / 2**k for k in range(n)]
        counts = cudaq.sample(qft, n, angles, shots_count=shots)
        ok = sum(counts.values()) == shots
    else:
        rng = random.Random(seed)
        thetas = [rng.uniform(0, 2 * math.pi) for _ in range(layers * n * 2)]
        counts = cudaq.sample(random_layers, n, layers, thetas, shots_count=shots)
        ok = sum(counts.values()) == shots
    return ok


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return ""


def memory_limit_bytes(tname):
    """Memory the state vector must fit in: GPU memory for nvidia targets, host RAM for qpp-cpu."""
    try:
        if tname.startswith("nvidia"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip().splitlines()[0]) * 2**20  # MiB
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="qpp-cpu,nvidia,nvidia-fp64")
    p.add_argument("--circuits", default="ghz,qft,random")
    p.add_argument("--min", type=int, default=10)
    p.add_argument("--max", type=int, default=34)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--time-cap", type=float, default=90.0,
                   help="stop growing n for a (target, circuit) once one run exceeds this many seconds")
    p.add_argument("--out", default=os.environ.get("SAKURA_ARTIFACT_DIR", "results"))
    p.add_argument("--tag", default="scaling")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}.csv")
    fields = ["target", "circuit", "qubits", "seconds", "status", "check_ok",
              "state_bytes", "cpu_count", "gpu"]
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    gpu = gpu_name()

    for tname in args.targets.split(","):
        try:
            set_target(tname)
        except Exception as e:  # target not available on this machine (e.g. no GPU)
            print(f"[skip] target {tname}: {e}", flush=True)
            continue
        bpa = TARGETS[tname][2]
        limit = memory_limit_bytes(tname) if bpa else None
        for circuit in args.circuits.split(","):
            run_circuit(circuit, 4, 10, 2, 0)  # warm-up: JIT compile outside the timed region
            for n in range(args.min, args.max + 1, args.step):
                row = dict(target=tname, circuit=circuit, qubits=n, cpu_count=os.cpu_count(),
                           gpu=gpu, state_bytes=(2**n) * bpa if bpa else "")
                # An out-of-memory allocation can abort the whole process instead of raising,
                # so record the memory wall without attempting it.
                if limit and row["state_bytes"] > 0.9 * limit:
                    row.update(seconds="", check_ok="",
                               status=f"skipped: state {row['state_bytes'] / 2**30:.0f} GiB > memory {limit / 2**30:.0f} GiB")
                    w.writerow(row)
                    f.flush()
                    print(f"{tname:12s} {circuit:7s} n={n:3d} {row['status']}", flush=True)
                    break
                t0 = time.perf_counter()
                try:
                    ok = run_circuit(circuit, n, args.shots, args.layers, seed=n)
                    row.update(seconds=round(time.perf_counter() - t0, 4), status="ok", check_ok=ok)
                except Exception as e:  # most often: out of GPU/host memory
                    row.update(seconds="", status=f"error: {type(e).__name__}: {str(e)[:120]}", check_ok="")
                w.writerow(row)
                f.flush()
                print(f"{tname:12s} {circuit:7s} n={n:3d} {row['seconds']!s:>10s}s {row['status'][:60]}",
                      flush=True)
                if row["status"] != "ok" or row["seconds"] > args.time_cap:
                    break
    f.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
