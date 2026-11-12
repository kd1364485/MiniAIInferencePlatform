# OnnxLearn — Phase 5: PyTorch / ONNX Runtime 推論

ロードマップ Phase 5 の学習記録。
PyTorchでモデル作成 → ONNX形式で書き出し → ONNX Runtimeで推論 を体験する。

## 環境

- Python 3.12.3 (venv: `~/OnnxLearn/venv`)
- torch 2.14.1+cu130 (CUDA有効)
- onnxruntime-gpu 1.30.0
- onnx 1.23.2 / onnxscript (export用)
- GPU: RTX 3060 (WSL2)

## ファイル

| ファイル | 内容 |
|---|---|
| export_onnx.py | TinyNet(Linear 4→3)を作り model.onnx へ書き出す |
| infer.py | model.onnx を ONNX Runtime で推論し、手計算matmulと照合 |
| model.onnx | 書き出されたモデル(計算グラフ + 重み) |

## 2026/10/09 M1＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### やったこと

PyTorchで最小モデル(`Linear(4→3)`)を作り、`torch.onnx.export` で `model.onnx` に書き出した。
ONNX Runtime (CUDAExecutionProvider) で推論し、ONNXファイルから重みW,bを読み出して
手計算 `x @ W.T + b` と照合した。max diff = 1.2e-7（一致）。

### 学んだこと

- `nn.Module`：PyTorchの全モデルの親クラス。「層を持つ箱」
nn.Linear(4, 3)：全結合層（fully connected layer）。内部に W(3×4行列)と b(3次元)を持つ
forward(x)：「入力xを受け取ったら何を返すか」の定義＝計算の流れ
数式で書くと y = x @ W.T + b（Phase 4のmatmulと同じ形）

- **ONNX = モデルの交換フォーマット**。PyTorch以外の環境でも読める汎用形式
- `torch.onnx.export` は「ダミー入力を1回流して計算グラフを記録する」方式（トレース）
- `dynamic_axes` でバッチ次元を可変にできる
- `InferenceSession(..., providers=[CUDA, CPU])` で実行バックエンドを選ぶ
- `sess.run(["output"], {"input": x})` は名前指定の入出力
- ONNXファイルの中身は protobuf の TensorProto（バイト列+shape）で、
  `onnx.numpy_helper.to_array` で numpy配列に変換して計算に使える
- **Linear層 = 行列積 + バイアス**（Phase 4のmatmulそのもの）
  「推論」= 学習済み重みの行列群に入力を通す計算

### つまずき

- pip/python が未インストール → `sudo apt install python3-pip python3-venv`
- torch 2.14 の新 exporter は `onnxscript` パッケージ必須（ModuleNotFoundErrorで判明）
- `python -c "..."` のダブルクォートが PowerShell 経由で崩れる → `\"` でエスケープ
- opset_version=17 を指定したが自動で18系に変換された（警告のみ、動作OK）

## 2026/10/09 M2＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

### やったこと

3層MLP(1024→2048→2048→10, 約6.3Mパラメータ)を`mlp.onnx`に書き出し、
ONNX Runtimeの CPU provider と CUDA provider で `sess.run` を500回ずつ計測した。
ウォームアップ50回を入れ、p50/p95/p99と推論/秒を比較。バッチ1とバッチ256で測定。

### 学んだこと

- バッチ1: CPU 2,711/s vs CUDA 4,247/s (x1.6)
  → 1往復の固定費(転送+カーネル起動)が支配的。vec_addと同じ構造
- バッチ256: CPU 21,504/s vs CUDA 299,264/s (x14)
  → 固定費は同じまま計算だけ増えるのでGPUが伸びる。
    CPUはx8しか伸びないのにGPUはx70伸びた
- **バッチ = 演算強度の別名**。固定費÷仕事量の比率がGPUの向きを決める
- `InferenceSession`に1 providerだけ渡しても、失敗すると**静かにCPUフォールバック**する
  → 「本当にそのproviderで動いたか」は実行ログ/確認が必須
- `nn.Sequential`で層を順繋ぎ、`nn.ReLU`は活性化関数(max(0,x)、matmulの退化防止)
- `os.environ.get`で環境変数からパラメータを取れる(BATCH=256 python ...)

### つまずき

- `libcublasLt.so.13 not found` でCUDA providerがロード失敗→CPUにフォールバックしていた
  → torch同梱の `nvidia/cu13/lib` と `nvidia/cudnn/lib` を `LD_LIBRARY_PATH` に追加して解決
- 「CUDA」を指定したつもりが実はCPUで計測していた(最初のrun)

## 2026/10/09 M3

### やったこと

MNIST(手書き数字)を MLP(784→256→128→10) で1エポック学習し `mnist.onnx` に書き出した。
Pythonの infer_mnist.py でテスト画像10枚を推論→9/10正解を確認した。

### 学んだこと

- 学習の5ステップ: `zero_grad` → `forward` → `loss` → `backward` → `opt.step`
- 推論は「重み固定でforward1回だけ」。更新ループは不要
- `DataLoader`がバッチ供給、`CrossEntropyLoss`が分類誤差、`Adam`が最適化
- ONNX出力はlogits(生スコア)。`argmax`=予測、softmax(`e^x/Σe^x`)=確信度
- `nn.Flatten`で(1,28,28)→784次元に変換

### つまずき

- 特になし

## 2026/10/09 M4

### やったこと

同じ `mnist.onnx` を **C++のONNX Runtime** で読み、`infer.cpp` で推論。
MNISTのIDXバイナリを `ifstream` で直接パースし、10枚で9/10正解
（Pythonと同じ結果・同じ誤り。ONNXの言語中立性を確認）。

### 学んだこと

- `Ort::Env`(ランタイム環境) → `Ort::Session`(読み込み済みモデル) → `session.Run`
- `Ort::Value::CreateTensor<float>(mem, ptr, size, shape, ndim)` でテンソル化
- `GetTensorMutableData<float>()` で出力の生配列を取り出す
- `std::max_element(...) - logits` = ポインタ引き算でindexを得る
  （ポインタの引き算は「型サイズで割った要素数」を返す言語仕様）
- IDXファイルはビッグエンディアン → `read_be32` でシフト+ORで32bit復元
  (`b[0]<<24 | b[1]<<16 | b[2]<<8 | b[3]`)

### つまずき

- ONNX RuntimeのC++用ライブラリは別途DL (`ort/` に展開、-I/-Lで指定)

## 2026/10/09 Phase5口頭チェック

### やったこと

Phase 5の5問口頭チェック。ONNXの目的、学習/推論の違い、provider確認、
logits/argmax/softmax、ポインタ引き算について説明。

### 学んだこと

- 「ONNX=交換フォーマット」「推論=重み固定」「argmax=index」等、
  個々の概念は分かるが「なぜその設計か」の言語化にまだ課題
- providerフォールバックは「確認する手段を覚える」が大事
  （get_providers / 実行ログ / nvidia-smi）

### つまずき

- Q1(ONNXの存在意義)が初回答えられなかった→exportの動機を覚える
- Q3で「8080ポート確認」とTCPの知識と混同した

### Q1. ONNXとは何か、なぜexportするか

- 初回: 分かりません → 再テスト: 正解
- **答え**: ONNX = Open Neural Network Exchange。モデルの交換フォーマット
  （計算グラフ + 重みをprotobufで保存）。PyTorch以外(C++, Java等)でも読める
  汎用形式。PyTorchのままだと「PyTorch + Python環境が必要」になるが、
  ONNX化すれば「推論専用の軽量エンジン(ONNX Runtime)」だけで動く。
  **学習と推論を分離するため**。

### Q2. 学習と推論の違い

- 初回: 正解
- **答え**: 学習は「重みを調整するループ」(forward→loss→backward→step)。
  推論は「学習済みの固定重みに1回通すだけ」。勾配・更新は不要。
  推論でループするのはリクエスト毎の呼び出しであって、重み調整ではない。

### Q3. providerフォールバックの罠

- 初回: フォールバックの理解は○、確認方法が×(8080ポートと混同)
- **答え**: `providers=["CUDAExecutionProvider"]` は「希望」。失敗時は
  **静かにCPUへフォールバック**する。確認方法:
  - セッション作成時の警告/エラーログを見る
  - `sess.get_providers()` で実際のproviderを確認
  - `nvidia-smi` でGPU使用率を見る

### Q4. logits / argmax / softmax

- 初回: △ → 再テスト: 正解
- **答え**:
  - logits = 10個の生スコア(未正規化の「各数字っぽさ」)
  - argmax = 「最大値が何番目か」を返す → そのindexが予測数字(0-9)
  - softmax = `e^x[i] / Σe^x[j]` で10個のスコアを合計1の確率に変換 → 確信度

### Q5. `it - logits` の型

- 初回: 正解
- **答え**: C++のポインタ引き算は「同じ型同士の場合、その型サイズで
  アドレス差を割った値（=要素数）」を返す。`float* - float*` なら
  `差バイト数 / sizeof(float)`。型が違うor別配列同士の引き算は未定義。

### 結果

```
Q1 ×→再○  Q2 ○  Q3 △  Q4 △→再○  Q5 ○
Phase 5 (PyTorch/ONNX推論) 完了。
次は Phase 6: C++サーバ + ONNX推論の融合 (Mini AI Inference Platform)
```




## 2026/11/04 再測定(M2 の CPU vs CUDA)

### やったこと

`bench_infer.py`(`mlp.onnx`、3 層 MLP 1024→2048→2048→10)を、変更せずに、バッチ 1 と 256 で **3 回ずつ**測り直した(各回ウォームアップ 50 回 + 計測 500 回)。
実行前に `sess.get_providers()` を確認し、CUDA が実際に使われていること(`['CUDAExecutionProvider', 'CPUExecutionProvider']`)を確かめた。環境は上の「環境」のとおり(ONNX Runtime 1.30.0、RTX 3060 Laptop、WSL2)。

| バッチ | 実行回数/秒 | p50 | p99 | 3 回の範囲(実行回数/秒) |
|---|---|---|---|---|
| 1 | CPU 3,752 / CUDA 4,705 | CPU 232 us / CUDA 191 us | CPU 547–681 us / CUDA 514–869 us | CPU 3,370–4,266 / CUDA 3,514–5,408 |
| 256 | CPU 92 / CUDA 1,247 | CPU 10.99 ms / CUDA 0.78 ms | CPU 11.9–13.1 ms / CUDA 1.37–1.40 ms | CPU 91–93 / CUDA 1,224–1,253 |

バッチ 256 の「サンプル/秒」(実行回数 × 256): CPU 約 23,600 / CUDA 約 318,000(**約 13.5 倍**)。上の M2 に書いた 21,504 と 299,264 は、この「サンプル/秒」の値。

### 結果の読み方

- **バッチ 256 は、前回とほぼ同じ結果で再現した**(CUDA が約 13〜14 倍)。3 回の揺れも小さい(CUDA 1,224–1,253 回/秒)。
- **バッチ 1 は、「CUDA が x1.6」とは言い切れなかった。** 3 回で CUDA/CPU は x1.25、x0.82(CUDA が負けた)、x1.60 と揺れた。
  1 回あたり 200 us 前後と短く、CPU 側も 1 スレッドあたりの揺れが大きいので、「GPU が少し速い、または同程度」が正確な言い方。
  M2 の「バッチ 1: x1.6」は、3 回のうちの良い回の値に近く、代表値としては甘かった。
- p50 で見ると、バッチ 1 は CPU 208–270 us、CUDA 170–244 us で、範囲が重なる。固定費(転送・カーネル起動)が計算より大きい、という説明は変わらない。
- バッチ 1 の CUDA は、入力のコピー(CPU → GPU)と出力のコピー(GPU → CPU)を、毎回 `sess.run` の中で行っている。この測定にはそれが含まれる(`IOBinding` で GPU 上に置けば、転送を除いた時間も測れるが、未実施)。

### 再現するときの注意

CUDA provider は、torch に同梱された CUDA ライブラリと、WSL 用のドライバのライブラリを探せる必要がある。次のように `LD_LIBRARY_PATH` を設定した。

```bash
SP=$(./venv/bin/python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib
BATCH=256 ./venv/bin/python bench_infer.py
```

`/usr/lib/wsl/lib` を先頭に置く理由: この PC では、Linux 版の `libcuda.so`(`/lib/x86_64-linux-gnu`)が先に読まれると「no CUDA-capable device」になる場合があったため
([phase4_CudaLearn](../phase4_CudaLearn/README.md) の表 5 の注意も参照)。
### 新しい venv で、README のとおりに再現できるか(2026/11/04)

新しい venv を作り直し、バージョンを固定して入れた(`torch==2.14.1 torchvision==0.29.1 onnx==1.23.2 onnxscript==0.7.2 onnxruntime-gpu==1.30.0 numpy==2.5.3`)。
ポートフォリオ内のスクリプトだけをコピーして、上から順に実行した。結果:

| 手順 | 結果 |
|---|---|
| `export_onnx.py` → `infer.py` | 再現した。ONNX の出力と手計算の差は 6e-8(M1 の 1.2e-7 と同程度の一致)。provider は CUDA |
| `export_mlp.py` | 再現した(6.32M パラメータ) |
| `bench_infer.py`(バッチ 1 / 256) | 再現した。バッチ 256 は CUDA が約 13 倍、バッチ 1 は同程度(上の再測定と同じ傾向) |
| `train_mnist.py` → `infer_mnist.py` | 再現した。テスト精度 93.6%、10 枚中 9 枚正解(M3 と同じ) |

確認していないこと: C++ 版(`infer.cpp`)は、ONNX Runtime の C++ ライブラリ(`ort/`)を別に用意する必要があり、今回はビルドしていない。
MNIST のデータは、通信を避けるため、既存の `data/` をコピーして使った(スクリプトは `download=True` で、無ければ自動でダウンロードする)。

**再現するときの落とし穴(実際に確認した)**: `LD_LIBRARY_PATH` に、torch に同梱された CUDA 13 と cuDNN のライブラリを入れないと、
`Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*` という警告が出るだけで、**黙って CPU で動く**
(`get_providers()` が `['CPUExecutionProvider']` だけになる)。CUDA と書いてあっても、必ず `get_providers()` で確かめること。