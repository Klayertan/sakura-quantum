# 作業ログ（ブログ素材）

## 2026-09-24 環境調査
- 高火力 DOK 料金（税込）：v100-32gb ¥0.016/秒（¥57.6/時）、h100-80gb ¥0.28/秒（¥1,008/時, 10 vCPU, 192GB RAM）、h100-8gpu-80gb ¥0.83/秒
- H100 は 1タスク最低60秒課金。初回¥3,000の無料クレジット（期限なし）
- V100 プランは 2027/3/31 終了予定（新規タスクは 2027/2/28 まで）
- 成果物は `/opt/artifact`（環境変数 `SAKURA_ARTIFACT_DIR`）に保存するとタスク終了後にダウンロード可能

## ハマりどころ
- CUDA-Q のドキュメントサイト（nvidia.github.io/cudaqx）が一部 404 → PyPI の cudaq 0.16.0.post1 / cudaq-solvers 0.6.0 を実際に入れて API を確認
- WSL の Ubuntu 24.04 で `python3 -m venv` が失敗（ensurepip なし）→ `sudo apt install python3.12-venv` か、`--without-pip` + get-pip.py で回避
- ローカル PC（dynabook）には NVIDIA GPU も Docker もない → CPU ターゲット `qpp-cpu` で動作確認、GPU は DOK で初実行

- **`pip install cudaq` の罠**：`cudaq` 0.16 は CUDA 13 版（cuda-quantum-cu13）を入れるが、cudaq-solvers 0.6.0 は cuda-quantum-cu12==0.14.* を要求 → 両方入って `cudaq` モジュールが二重に。requirements.txt を cu12 版に固定（cuda-quantum-cu12==0.14.2, cudaq-solvers-cu12==0.6.0, pyscf==2.14.0）。依存解決がしばらく止まって見えたのもこれが原因
- `import cudaq_solvers` で `libgfortran.so.5` がないエラー → wheel に同梱されていない。Dockerfile に `libgfortran5` を追加
- CUDA-Q 0.14 の `SampleResult` には `.keys()` がない → `.items()` を使う
- PySCF の結果は numpy.bool_ なので `json.dump` できない → `bool()` で変換
- `create_molecule` が作業ディレクトリに `H 0-pyscf.chk` などのファイルを書き出す → .gitignore に追加
- GPU の OOM はプロセスごと落ちる可能性 → メモリに入らない量子ビット数は実行せず「skipped」として記録するガードを追加
- **VQE の反復回数**：COBYLA の既定 tol=1e-12・上限なしだと LiH（12 qubit, 631 項, 92 パラメータ）が CPU で終わらない。1 回の observe ≈ 4 秒、100 反復で 369 秒・誤差 20 mHa。L-BFGS（parameter shift で 1 勾配 = 184 回評価）は 20 反復が 18 分で終わらず → `--max-iterations`（既定 3000）を追加、CPU vs GPU 比較は 20 反復固定で秒/反復を比べる
- 分子サイズ（sto-3g）：H2 4q/15項/3パラメータ、LiH 12q/631項/92、N2 CAS(6,6) 12q/411項/117、BeH2 14q/674項/204

## 結果メモ
### スモークテスト（4コア Linux, GPU なし, CUDA-Q 0.14.2）
- qpp-cpu 20 qubit：GHZ 1.8 s、QFT 17.8 s、random(10層) 49.5 s。16→20 qubit でランダム回路は約 23 倍
- H2 VQE -1.137176 Ha（FCI との差 4.9e-10, 44 反復, 7.8 s）
- H2 解離曲線 d=2.5Å：HF -0.7029 / FCI -0.9361（差 0.23 Ha）、VQE は FCI と一致

（DOK 実行後に追記）
