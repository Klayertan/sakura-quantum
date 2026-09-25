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

- GitHub Actions のログイン失敗（unauthorized）：レジストリのユーザー名は `klayertan` なのに Secret を `klayer` にしていた。レジストリ名（ホスト名）とユーザー名は別物
- DOK のタスク作成画面は「エントリーポイント」と「コマンド」が別欄。コマンドは引数だけ渡す（`smoke`）
- **V100 で segfault**：qpp-cpu は完走したが `nvidia` ターゲットで Segmentation fault。cuStateVec は v1.9 で Volta（sm_70）サポート終了、同梱は 1.15 → GPU シミュレーションは H100 のみ
- **`libcudaq-solvers.so: cannot enable executable stack`**：python:3.12-slim が Debian 13（glibc 2.41）になり、実行可能スタックを要求する .so の読み込みを拒否。`readelf -lW` で GNU_STACK が RWE なのは libcudaq-solvers.so だけ。→ `python:3.12-slim-bookworm` に固定
- run.sh が常に exit 0 だったので、GitHub Actions のスモークテストが失敗を見逃していた → 失敗したステップを記録して最後に exit 1

## 結果メモ
### DOK 初回 smoke（v100-32gb, 2026-09-25）
- 4 vCPU / メモリ 54GB。qpp-cpu 20 qubit：GHZ 3.8 s、QFT 38.2 s、random 108.7 s（手元の 4 コアより約 2 倍遅い）
- GPU ターゲットは segfault、VQE は ImportError（上記）

### DOK smoke（h100-80gb, 2026-09-25, 2 回実行）
- 20 vCPU / メモリ 216GB、`GPUs visible: 1`。実行時間 6 分 35 秒（約 110 円）
- 20 qubit の CPU（qpp-cpu, 20 vCPU）vs GPU（nvidia fp32）:
  - GHZ 0.95 s → 0.038 s（約 25 倍）
  - QFT 8.13 s → 0.14 s（約 58 倍）/ 2 回目 8.16 s → 0.11 s（約 75 倍）
  - random(10層) 17.6 s → 0.31 s（約 57 倍）/ 2 回目 16.0 s → 0.34 s（約 47 倍）
  - GPU は 4〜20 qubit でほぼ横ばい（0.03〜0.3 s）＝この規模ではオーバーヘッドが支配的
- H2 VQE（qpp-cpu）：-1.137176 Ha、44 反復、23 s（手元 4 コアの 7.8 s より遅い。小さな問題はコア数より 1 コアの速さ）
- **LiH VQE（nvidia-fp64）：94 評価で 182 s = 1.94 s/評価**。CPU（4 コア手元）の 3.7 s/評価とあまり変わらない。12 qubit では GPU の強みが出ない → 第 3 回のネタ
  - エネルギー -7.862027 Ha は手元 CPU の 100 反復の結果と完全一致（再現性 OK）
  - `--max-iterations 20` でも COBYLA は「パラメータ数 + 2」回（= 94）までは評価する（`Invalid MAXFUN` 警告）
- 同じ smoke を 2 回実行：LiH 182.05 s / 182.80 s と、ばらつきはごく小さい
- ミス：コピーして新規作成した時にコマンドを `smoke` のまま実行してしまった（約 110 円）。コピー時はコマンド欄を必ず確認

### スモークテスト（4コア Linux, GPU なし, CUDA-Q 0.14.2）
- qpp-cpu 20 qubit：GHZ 1.8 s、QFT 17.8 s、random(10層) 49.5 s。16→20 qubit でランダム回路は約 23 倍
- H2 VQE -1.137176 Ha（FCI との差 4.9e-10, 44 反復, 7.8 s）
- H2 解離曲線 d=2.5Å：HF -0.7029 / FCI -0.9361（差 0.23 Ha）、VQE は FCI と一致

（DOK 実行後に追記）

### DOK bench 1 回目（h100-80gb, 2026-09-25 18:11, 約 11 分・約 190 円）
- qpp-cpu は 90 秒上限で GHZ 27 / QFT 23 / random 22 qubit まで
- **GPU は state が 32 GiB になった所で segfault**：fp32 n=32、fp64 n=31。GHZ で落ちたせいで同じプロセスの QFT・random が走らなかった
- nvidia-smi：H100 80GB HBM3、81559 MiB、MIG 無効、ドライバ 580.173.02（CUDA 13.0）→ GPU が小さいわけではない
- → サンプリングで state の約 2.5 倍の作業メモリが要ると判断。ガードを「GPU メモリの 40%」に変更、(target, circuit) ごとに別プロセス、kernel から明示的な mz() を削除
- tensornet は起動直後に `CUTENSORNET_STATUS_INVALID_VALUE in line 377` で abort（未解決）

### DOK bench 2 回目（h100-80gb, 2026-09-25 19:03, 約 14 分・約 240 円）
- 全 GPU スイープが segfault なしで完走し、メモリ上限で「skipped」と記録
- **実用上の上限：fp32 31 qubit / fp64 30 qubit**（理論上は 80GB に 33 / 32 qubit 入るが、サンプリング用の作業領域が必要）
- CPU で測れた最大の n での比較（CPU 20 vCPU vs H100）:
  - GHZ 27 qubit：CPU 141.9 s → fp32 0.040 s（**約 3,600 倍**）/ fp64 0.049 s（約 2,900 倍）
  - QFT 24 qubit：CPU 193.2 s → fp32 0.223 s（約 870 倍）/ fp64 0.171 s（約 1,130 倍）
  - random 22 qubit：CPU 90.1 s → fp32 0.207 s（約 430 倍）/ fp64 0.310 s（約 290 倍）
- GPU の最大サイズでも 1 秒前後：fp32 31 qubit で GHZ 0.23 s / QFT 0.73 s / random 1.08 s
- fp64 は fp32 とほぼ同じ速さ（H100 は倍精度も速い）
- tensornet は同じエラーで abort → `bench` から外して別ジョブ `tensornet` に
- データ：`blog/data/h100-bench-2026-09-25/`、グラフ：`blog/images/scaling_time.png`, `speedup.png`

