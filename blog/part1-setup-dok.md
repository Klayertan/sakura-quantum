# さくらのGPUで量子コンピュータを「作る」：高火力DOK入門【量子シミュレーション第1回】

量子コンピュータ、使ってみたいけど手元にない。実機のクラウドは順番待ちで、結果にもノイズが乗る。
それなら **GPU の上に量子コンピュータを丸ごと再現してしまおう**、というのがこのシリーズです。

使うのはさくらインターネットのコンテナ型 GPU クラウド **高火力 DOK**。NVIDIA H100 を秒単位課金で借りて、NVIDIA の量子シミュレーション SDK **CUDA-Q** を動かします。

**シリーズ構成**
1. **第1回（この記事）**：量子シミュレーションの仕組みと、高火力 DOK で動かすまで
2. 第2回：H100 で何量子ビットまで行ける？CPU vs GPU ベンチマーク
3. 第3回：量子化学を GPU で。VQE で分子のエネルギーを計算する

**こんな人向け**
- 量子コンピュータに興味はあるけれど、まだ触ったことがないエンジニア
- 研究で GPU が必要で、さくらの GPU を借りるとどうなるか知りたい人

コード一式は GitHub で公開しています：【TODO: リポジトリURL】

---

## 1. GPU で量子コンピュータが「作れる」理由

### 量子ビットは「確率の振幅」のリスト

古典ビットは 0 か 1 のどちらかです。量子ビットは 0 と 1 の**重ね合わせ**で、状態を2つの複素数（振幅）で表します。

```
|ψ⟩ = α|0⟩ + β|1⟩     （|α|² が 0 が出る確率、|β|² が 1 が出る確率）
```

量子ビットが n 個になると、`00…0` から `11…1` までの **2ⁿ 通りすべてに振幅が1つずつ**付きます。これを**状態ベクトル**と呼びます。

| 量子ビット数 | 振幅の数 |
|---|---|
| 2 | 4 |
| 10 | 1,024 |
| 30 | 約10.7億 |
| 40 | 約1.1兆 |

量子ゲート（量子コンピュータの命令）は、この巨大なベクトルに**行列を掛ける操作**です。つまり、量子コンピュータをシミュレーションするとは「巨大なベクトルに小さな行列を何度も掛ける」ことです。これはまさに **GPU が最も得意な仕事**です。

### 1量子ビット増えるごとにメモリは2倍

振幅1つは複素数なので、単精度（complex64）で 8 バイト、倍精度（complex128）で 16 バイト必要です。

![量子ビット数と状態ベクトルのメモリ](images/memory_wall.png)

| メモリ | 単精度で扱える量子ビット数 | 倍精度で扱える量子ビット数 |
|---|---|---|
| ノートPC 16GB | 30 | 29 |
| V100 32GB | 31 | 30 |
| **H100 80GB** | **33** | **32** |
| H100×8 = 640GB | 36 | 35 |

1量子ビット増やすたびに必要なメモリが倍になる。これが量子コンピュータのシミュレーションが難しい理由であり、同時に**本物の量子コンピュータが強い理由**でもあります。
（第2回では、この壁を「テンソルネットワーク」という別の手法ですり抜けて、100量子ビットに挑戦します。）

### CUDA-Q とは

[CUDA-Q](https://developer.nvidia.com/cuda-q) は NVIDIA のオープンソース量子プログラミング環境です。Python で量子回路を書き、**「ターゲット」を切り替えるだけで**同じコードを CPU でも GPU でも実行できます。

```python
import cudaq

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)       # n 量子ビットを用意
    h(q[0])                    # 1つ目を重ね合わせに
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1]) # CNOT で次々に「もつれ」させる
    mz(q)                      # 測定

cudaq.set_target("qpp-cpu")    # CPU で実行
print(cudaq.sample(ghz, 20))

cudaq.set_target("nvidia")     # GPU で実行（コードはそのまま）
print(cudaq.sample(ghz, 20))
```

この GHZ 状態は「全部 0」か「全部 1」しか出ない、量子もつれの代表例です。結果が `{ 000…0: 約500, 111…1: 約500 }` になれば、シミュレータが正しく動いています。

---

## 2. 高火力 DOK とは

[高火力 DOK](https://www.sakura.ad.jp/koukaryoku-dok/) は、**Docker イメージを渡すと GPU 上で実行してくれる**サービスです。サーバーの構築や SSH は不要で、使った秒数だけ課金されます。

| プラン | GPU | vCPU / メモリ | 料金（税込） |
|---|---|---|---|
| v100-32gb | V100 32GB | 3 / 40GB | ¥0.016/秒（¥57.6/時） |
| h100-80gb | H100 80GB | 10 / 192GB | ¥0.28/秒（¥1,008/時） |
| h100-8gpu-80gb | H100×8 | 160 / 1,760GB | ¥0.83/秒 |

※ 2026年9月時点。H100 プランは1タスクあたり最低60秒の課金。V100 プランは 2027年3月末で提供終了予定。

**初回は ¥3,000 分の無料クレジット**が付くので、この記事の内容は無料枠でも十分試せます。

研究用途で嬉しいポイント：
- **バッチ処理向き**：「スクリプトを投げて、結果ファイルを受け取る」だけ。消し忘れによる課金事故がない
- **秒課金**：数分で終わる実験を何度も回すのに向いている
- **国内リージョン**：データを国外に出せない研究でも使いやすい

---

## 3. 実験用コンテナを作る

### ファイル構成

```
sakura-quantum/
  Dockerfile
  requirements.txt      # cudaq, cudaq-solvers, pyscf
  run.sh                # エントリポイント（smoke / bench / vqe / all）
  bench/scaling.py      # 第2回のベンチマーク
  vqe/vqe_molecules.py  # 第3回の量子化学計算
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run.sh /app/run.sh
COPY bench /app/bench
COPY vqe /app/vqe
RUN chmod +x /app/run.sh

WORKDIR /app
ENTRYPOINT ["/app/run.sh"]
CMD ["all"]
```

ポイントは、**CUDA 入りの重いベースイメージが不要**なことです。`pip install cudaq` で CUDA ランタイムも一緒に入り、GPU ドライバは DOK のホスト側から提供されます。

### 結果は `/opt/artifact` に書く

DOK では、タスク内で環境変数 `SAKURA_ARTIFACT_DIR`（= `/opt/artifact`）に保存したファイルを、終了後にコントロールパネルからダウンロードできます。`run.sh` では出力先をこの変数で決めています。

```bash
OUT="${SAKURA_ARTIFACT_DIR:-$PWD/results}"   # DOK 上なら /opt/artifact、ローカルなら ./results
```

---

## 4. まずはローカル（CPU）で動作確認

GPU の時間を無駄にしないよう、まず手元の PC で小さく動かします。筆者の環境は GPU なしの Windows ノートなので、WSL（Ubuntu 24.04）を使いました。

```bash
python3 -m venv ~/cq
~/cq/bin/pip install -r requirements.txt
PATH=~/cq/bin:$PATH ./run.sh smoke
```

> **ハマりどころ**：Ubuntu 24.04 の WSL では `python3 -m venv` が `ensurepip is not available` で失敗することがあります。`sudo apt install python3.12-venv` を入れれば解決します。

実行結果：

```
【TODO: スモークテストの出力を貼る】
```

GHZ 回路の結果が「全部 0」と「全部 1」だけになっていること、H2 分子の VQE エネルギーが厳密解（約 -1.137 Hartree）と一致していることを確認できました。

---

## 5. 高火力 DOK で実行する

### ① イメージをレジストリに push

```bash
docker build -t <レジストリ>/sakura-quantum:latest .
docker push <レジストリ>/sakura-quantum:latest
```

レジストリは、さくらのクラウドの「コンテナレジストリ」（`xxx.sakuracr.jp`）か Docker Hub などが使えます。

【TODO: コンテナレジストリ作成画面のスクリーンショット】

### ② タスクを作成

コントロールパネルの「タスク」→「作成」で、以下を設定します。

| 項目 | 設定値 |
|---|---|
| イメージ | `<レジストリ>/sakura-quantum:latest` |
| プラン | まずは `v100-32gb`（安い）で `smoke`、本番は `h100-80gb` |
| コマンド | `/app/run.sh smoke`（本番は `/app/run.sh all`） |
| レジストリ認証 | プライベートレジストリならユーザー名・パスワード |

【TODO: タスク作成画面のスクリーンショット】

### ③ 結果をダウンロード

タスクが完了すると、「アーティファクト」から `/opt/artifact` の中身を zip でダウンロードできます。`nvidia-smi.txt` に GPU の情報が、`log.txt` に実行ログが入っています。

```
【TODO: DOK 上での smoke 実行ログ（GPU 名、実行時間）】
```

かかった時間は【TODO】秒、料金は約【TODO】円でした。

---

## まとめ

- 量子コンピュータのシミュレーションは「2ⁿ 要素のベクトル × 行列」で、GPU の得意分野
- メモリは1量子ビットごとに2倍。H100 80GB なら約33量子ビットまで
- 高火力 DOK なら Docker イメージを渡すだけで H100 を秒単位で借りられる。無料クレジットで試せる

次回は、いよいよ H100 の本気を見ます。**CPU と GPU で何倍の差が出るのか、何量子ビットで限界が来るのか**を実測します。

---

**参考**
- [高火力 DOK（さくらインターネット）](https://www.sakura.ad.jp/koukaryoku-dok/)
- [NVIDIA CUDA-Q](https://developer.nvidia.com/cuda-q)
