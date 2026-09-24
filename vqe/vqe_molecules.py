"""UCCSD-VQE ground-state energies for small molecules with CUDA-Q Solvers.

Each result is checked against an exact classical reference (FCI, or CASCI
for active-space runs) computed independently with PySCF.
"""
import argparse
import json
import os
import time

import cudaq
import cudaq_solvers as solvers
from pyscf import fci, gto, mcscf, scf

CHEMICAL_ACCURACY = 1.6e-3  # Hartree (= 1 kcal/mol)

# name -> (geometry in Angstrom, active space (n_electrons, n_orbitals) or None for the full space)
MOLECULES = {
    "H2": ([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))], None),
    "LiH": ([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))], None),
    "BeH2": ([("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.3264)), ("H", (0.0, 0.0, -1.3264))], None),
    "N2": ([("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))], (6, 6)),
}

TARGETS = {
    "qpp-cpu": ("qpp-cpu", None),
    "nvidia": ("nvidia", "fp32"),
    "nvidia-fp64": ("nvidia", "fp64"),
}


def set_target(name):
    target, option = TARGETS[name]
    if option:
        cudaq.set_target(target, option=option)
    else:
        cudaq.set_target(target)


def reference_energies(geometry, active, basis="sto-3g"):
    """Hartree-Fock and exact (FCI / CASCI) energies from PySCF."""
    mol = gto.M(atom=[(a, xyz) for a, xyz in geometry], basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    if active:
        exact = mcscf.CASCI(mf, active[1], active[0]).kernel()[0]
    else:
        exact = fci.FCI(mf).kernel()[0]
    return mf.e_tot, exact


def run_vqe(geometry, active, optimizer, max_iterations, basis="sto-3g"):
    kwargs = dict(casci=False, verbose=False)
    if active:
        kwargs.update(nele_cas=active[0], norb_cas=active[1])
    molecule = solvers.create_molecule(geometry, basis, 0, 0, **kwargs)
    n_qubits = 2 * molecule.n_orbitals
    n_electrons = molecule.n_electrons
    spin = 0

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])  # Hartree-Fock reference state
        solvers.stateprep.uccsd(q, thetas, n_electrons, spin)

    n_params = solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits, spin)
    t0 = time.perf_counter()
    energy, params, history = solvers.vqe(ansatz, molecule.hamiltonian, [0.0] * n_params,
                                          optimizer=optimizer, max_iterations=max_iterations)
    seconds = time.perf_counter() - t0
    return dict(vqe_energy=energy, qubits=n_qubits, electrons=n_electrons,
                hamiltonian_terms=molecule.hamiltonian.term_count, parameters=n_params,
                iterations=len(history), hit_iteration_cap=len(history) >= max_iterations,
                seconds=round(seconds, 3), seconds_per_iteration=round(seconds / max(len(history), 1), 5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="nvidia-fp64")
    p.add_argument("--molecules", default="H2,LiH,BeH2,N2")
    p.add_argument("--optimizer", default="cobyla")
    p.add_argument("--max-iterations", type=int, default=3000,
                   help="optimizer iteration cap; COBYLA's default tolerance otherwise runs for hours "
                        "on LiH-sized problems (~4 s/iteration on a 4-core CPU)")
    p.add_argument("--h2-curve-points", type=int, default=15,
                   help="points on the H2 dissociation curve (0 to skip)")
    p.add_argument("--out", default=os.environ.get("SAKURA_ARTIFACT_DIR", "results"))
    p.add_argument("--tag", default="vqe")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {"molecules": [], "h2_curve": []}
    path = os.path.join(args.out, f"{args.tag}.json")

    def save():
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    for tname in args.targets.split(","):
        try:
            set_target(tname)
        except Exception as e:
            print(f"[skip] target {tname}: {e}", flush=True)
            continue
        for name in args.molecules.split(","):
            geometry, active = MOLECULES[name]
            hf, exact = reference_energies(geometry, active)
            r = run_vqe(geometry, active, args.optimizer, args.max_iterations)
            r.update(molecule=name, target=tname, hf_energy=hf, exact_energy=exact,
                     active_space=active, error=abs(r["vqe_energy"] - exact))
            r["chemically_accurate"] = bool(r["error"] < CHEMICAL_ACCURACY)
            results["molecules"].append(r)
            save()
            print(f"{tname:12s} {name:5s} q={r['qubits']:2d} params={r['parameters']:3d} "
                  f"VQE={r['vqe_energy']:.6f} exact={exact:.6f} err={r['error']:.2e} "
                  f"iters={r['iterations']} {r['seconds']}s", flush=True)

    # H2 dissociation curve on the last target: where HF fails and VQE keeps up with FCI
    n = args.h2_curve_points
    for k in range(n):
        d = 0.3 + (2.5 - 0.3) * k / max(n - 1, 1)
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, d))]
        hf, exact = reference_energies(geometry, None)
        r = run_vqe(geometry, None, args.optimizer, args.max_iterations)
        results["h2_curve"].append(dict(distance=round(d, 4), hf_energy=hf, exact_energy=exact,
                                        vqe_energy=r["vqe_energy"]))
        save()
        print(f"H2 d={d:.3f}  HF={hf:.6f} VQE={r['vqe_energy']:.6f} FCI={exact:.6f}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
