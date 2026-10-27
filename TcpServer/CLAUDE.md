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
**Phase 6 完了**(2026-10-27 時点。M1〜M5: 推論サーバー → 切り分け → 重いモデル測定 → dynamic batching → Request Queue)。次は Phase 7(TensorRT / GPU 最適化)。
Phase 6 に GPU は含めていない(Phase 7 で扱う)。未解決: サーバー内でだけ出る Run() の約 +70us の正体。
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
