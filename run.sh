#!/bin/bash
# Entrypoint: run.sh [smoke|bench|vqe|all]
# Results go to $SAKURA_ARTIFACT_DIR (set by 高火力 DOK, /opt/artifact) so they can be downloaded afterwards.
set -uo pipefail
cd "$(dirname "$0")"
OUT="${SAKURA_ARTIFACT_DIR:-$PWD/results}"
mkdir -p "$OUT"
JOB="${1:-all}"

exec > >(tee -a "$OUT/log.txt") 2>&1
echo "=== job=$JOB start $(date -Is) ==="
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || echo "no nvidia-smi (CPU only)"
python -c "import cudaq; print('cudaq', cudaq.__version__)"
nproc; free -g | head -2

bench() {
  python bench/scaling.py --targets qpp-cpu,nvidia,nvidia-fp64 --min 10 --max 36 --out "$OUT"
  # bonus: tensor-network backend on shallow circuits far beyond state-vector memory
  python bench/scaling.py --targets tensornet --circuits ghz,random --layers 2 \
    --min 20 --max 100 --step 10 --shots 100 --tag tensornet --out "$OUT"
}

vqe() {
  python vqe/vqe_molecules.py --targets nvidia-fp64 --molecules H2,LiH,N2,BeH2 --max-iterations 3000 --out "$OUT"
  # CPU vs GPU on the same VQE problem: a fixed 20 iterations, compared as seconds/iteration
  # (running the CPU to convergence would take hours of billed GPU-node time)
  python vqe/vqe_molecules.py --targets qpp-cpu,nvidia-fp64 --molecules LiH --max-iterations 20 \
    --h2-curve-points 0 --tag vqe_cpu_vs_gpu --out "$OUT"
}

case "$JOB" in
  smoke)
    python bench/scaling.py --targets qpp-cpu,nvidia --min 4 --max 20 --step 4 --tag smoke --out "$OUT"
    python vqe/vqe_molecules.py --targets qpp-cpu --molecules H2 --h2-curve-points 3 --tag smoke_vqe --out "$OUT"
    # GPU seconds/iteration on LiH, to size the full run before paying for it (skipped without a GPU)
    python vqe/vqe_molecules.py --targets nvidia-fp64 --molecules LiH --max-iterations 20 \
      --h2-curve-points 0 --tag smoke_vqe_gpu --out "$OUT"
    ;;
  bench) bench ;;
  vqe) vqe ;;
  all) bench; vqe ;;
  *) echo "usage: run.sh [smoke|bench|vqe|all]"; exit 2 ;;
esac
echo "=== job=$JOB end $(date -Is) ==="
