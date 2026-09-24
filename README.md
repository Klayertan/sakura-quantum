# sakura-quantum

Simulating a quantum computer on the Sakura Internet 高火力 DOK GPU cloud with NVIDIA CUDA-Q.

- `bench/scaling.py`: how many qubits, and how fast? GHZ / QFT / random circuits on CPU vs GPU (fp32/fp64), plus the tensor-network backend for 40–100 qubits.
- `vqe/vqe_molecules.py`: molecular ground-state energies (H2, LiH, BeH2, N2) with UCCSD-VQE, checked against exact PySCF results.
- `analysis/plot.py`: charts for the blog.

## Local smoke test (CPU, no GPU needed)
Linux / WSL:
```bash
python3 -m venv ~/cq && ~/cq/bin/pip install -r requirements.txt pandas matplotlib
~/cq/bin/python -m pip show cudaq | head -2
PATH=~/cq/bin:$PATH ./run.sh smoke        # writes results/
```

## Run on 高火力 DOK
1. Build and push the image to a registry DOK can pull from (Sakura container registry `<name>.sakuracr.jp`, or Docker Hub):
   ```bash
   docker build -t <registry>/sakura-quantum:latest .
   docker push <registry>/sakura-quantum:latest
   ```
2. In the DOK control panel, create a task:
   - Plan: `h100-80gb` (¥0.28/s) or `v100-32gb` (¥0.016/s, good for a first try)
   - Image: `<registry>/sakura-quantum:latest` (with registry credentials if private)
   - Command: `/app/run.sh all` (or `bench`, `vqe`, `smoke`)
3. When the task finishes, download the artifacts (`SAKURA_ARTIFACT_DIR` = `/opt/artifact`) and unzip them into `results/`.
4. Make the charts: `python analysis/plot.py --results results --out blog/images`

New accounts get ¥3,000 of free credit. The H100 plan bills a 60-second minimum per task.
