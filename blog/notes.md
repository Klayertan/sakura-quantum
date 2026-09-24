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

## 結果メモ
（スモークテスト / DOK 実行後に追記）
