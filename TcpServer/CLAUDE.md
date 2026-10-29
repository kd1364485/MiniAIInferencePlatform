# SystemProbe — プロジェクトコンテキスト

## 目的
約1年かけて **Mini AI Inference Platform**(C++ / Linux / Networking / Concurrency / CUDA / ONNX Runtime / TensorRT / Docker / Monitoring を統合)を作る、ML Systems ポートフォリオ計画の一部。
ML Research ではなく **ML Systems** が狙い。

`SystemProbe` は **Phase 1**(Modern C++ + Linux)の作品: Linux 上で CPU / Memory / Process / Thread / Network / Disk を取得する CLI。
学ぶこと: `/proc`、process、file、permission、CMake、RAII、smart pointer、error handling。

## 計画の原本(WSL からの参照パス)
- `/mnt/c/Users/seiya/SystemProbe/学習ロードマップ/統合ロードマップ.md`
- `/mnt/c/Users/seiya/SystemProbe/学習ロードマップ/実行していくうえでのルール.md`

## ロードマップ(固定)
1. Phase 1: Modern C++ + Linux Systems(SystemProbe)
2. Phase 2: Networking / TCP Server
3. Phase 3: Concurrency / Performance
4. Phase 4: CUDA
5. Phase 5: PyTorch / ONNX Runtime による Inference
6. Phase 6: C++ Systems + ML Inference 融合
7. Phase 7: TensorRT / GPU 最適化
8. Phase 8: Docker GPU / Monitoring / AI Infrastructure
9. Phase 9: Kubernetes / Distributed Inference

## 現在の進捗
**Phase 9 進行中**(2026-10-29 時点)。範囲は相談して決めた: 先に「分散の本質」を自作 → そのあと Kubernetes の最小構成。完了: M1 自作ロードバランサ `lb`(epoll の L4 プロキシ、ヘルスチェック、DNS で台数に追従)、M2 Docker に組み込み + 複数台の測定(1枚の GPU ではワーカーを増やすと遅くなる: 62k → 50k → 40k req/s、原因はバッチが割れること)。M3 Kubernetes(kind、Deployment/Service/probe/ローリング更新。kill -9 は約0.05秒で収まるが SIGSTOP は probe で約4.4秒、ローリング更新でも既存接続は切れる)の測定まで完了。残り: M3 のコード読み・口頭確認、Phase 9 の総括、main へのマージ(ユーザーの許可を取ってから)。
Phase 8 完了(2026-10-28。M1〜M5 + 総括: Docker + NVIDIA Container Toolkit → 2段階ビルドのイメージ → /metrics → Prometheus + Grafana → 資源・障害の見え方・再起動の検知)。
Phase 7 完了(2026-10-28。GPU は RTX 3060 Laptop 6GB。M1〜M4 + 総括: GPU 比較基盤 → TensorRT(FP16) → C++ サーバーに GPU(cuda/trt) → INT8)。Phase 6 完了(2026-10-27。M1〜M5: 推論サーバー → … → Request Queue)。
未解決: LB を挟むと直接より約4〜10%速く出た理由(揺らぎ約9%と近く、未確認)、複数プロセスが1枚の GPU を分け合うときの時分割のコスト(バッチが割れる影響と、切り分けていない)、`-p` で公開すると約33%遅い原因の確定(docker-proxy を無効にした再測定)、本物のクラッシュ・OOM での挙動、アラート(通知)は未実装、ORT 経由の TensorRT がネイティブより速かった理由。INT8 は C++ サーバー未対応。サーバーの外にあると分かった約1.3ms(カーネルのネットワーク・起床・bench 側)の内訳は未測定。
関連フォルダ: リポジトリ `~/seiya7183517/`(GitHub: `kd1364485/MiniAIInferencePlatform`)の `TcpServer/`。ホーム直下に `CudaLearn/`, `OnnxLearn/`。ビルド・実行手順は `TcpServer/README.md` の冒頭。

## 進め方のルール
1. **期間でなく到達条件**で次へ進む。
2. **新技術を増やしすぎない**(当面 Kubernetes / Terraform / 大規模 LLM 運用は対象外)。
3. 毎フェーズ「**作る → 測る → 問題を見つける → 改善する → 説明する**」まで行う。「動いた」だけで終わらせず、なぜ遅い/速くなったかを説明できる状態にする。
4. 毎日の開発時間を 100% にしない。1日の学び・重要事項を README に簡潔にまとめる時間を確保する。焦らない。
5. プログラムは AI エージェント(Claude Code)が書く(速度優先)。その代わり**コードの内容を理解・把握する**。分からない箇所は質問して潰す。

## Claude への依頼方針
- コードを書く/直す際は、**なぜそう書くのか・対応する OS 概念(process, virtual memory, fd, RAII 等)**を簡潔に説明する。
- 各フェーズの完了条件は「説明できること」。README の学びメモ作成を手伝う。
- 性能の話は必ず測定値(latency / throughput / p50 / p95 / p99 など)で語る。
- 計画外の新技術の導入は提案前に確認する。
