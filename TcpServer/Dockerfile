# syntax=docker/dockerfile:1
#
# Phase 8 M2: infer_server(C++ 推論サーバー)の GPU 版イメージ。2 段階ビルド。
#
# ビルド(TcpServer/ で実行):   docker build -t infer-server:gpu .
# 実行(GPU をコンテナに渡す):
#   docker run --rm --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" infer-server:gpu \
#       /models/mlp_dyn.onnx 2 1 0 32 1 cuda
#   引数は infer_server と同じ: <model> [workers] [intra_op] [per_worker] [max_batch] [infer_threads] [provider]
#
# 設計の要点:
#  - 2 段階: 「ビルド用」(g++ / cmake が要る)と「実行用」(CUDA のライブラリだけ)を分け、実行用イメージを小さくする
#  - GPU ドライバはイメージに入れない。NVIDIA Container Toolkit が、起動時にホストのドライバのライブラリを注入する
#    (イメージに入れるのは CUDA / cuDNN の「ライブラリ」まで)
#  - ONNX Runtime(GPU 版 C++ パッケージ)はビルド中に取得し、SHA-256 を照合する(236MB を git に入れない)
#  - モデルはイメージに入れず、実行時にマウントする(大きい・設定ごとに違う)
#  - 一般ユーザーで動かす(GPU を使うのに root は要らない)
#  - TensorRT は含めない(provider は cpu / cuda)。trt / trt-fp16 を指定すると起動時に例外で止まる(fail fast)

# ============ ビルド用の段階 ============
FROM ubuntu:24.04 AS build

# ONNX Runtime の GPU 版(CUDA 13)C++ パッケージ。SHA-256 は公式リリースが公開している値
ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=382d79133112388cf94ce5855789b7c9bef12bef76a08b6b277e5a317213adcd

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake make curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# ort-cache/ort.tgz(ホストで取得済みのファイル)があればそれを使い、無ければここで取得する。
#   ビルドの中のダウンロードは、この環境では途中で止まることがあった(curl: (56)/(28))。ホストで
#   curl -C -(途中から再開)で取っておけば確実。どちらの場合も、SHA-256 を照合してから使う。
#   取得する場合は、受信が止まったら早く諦めて再試行する:
#     --speed-time 60 --speed-limit 50000 : 60 秒間ずっと 50KB/s 未満なら切断
#     --retry 5 --retry-all-errors        : 切断・タイムアウトを含む失敗を最大 5 回やり直す
COPY ort-cache/ /tmp/ort-cache/
RUN if [ -f /tmp/ort-cache/ort.tgz ]; then cp /tmp/ort-cache/ort.tgz ort.tgz; \
    else curl -fsSL --connect-timeout 30 --speed-time 60 --speed-limit 50000 --retry 5 --retry-delay 5 --retry-all-errors -o ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz"; fi \
 && echo "${ORT_SHA256}  ort.tgz" | sha256sum -c - \
 && mkdir ort && tar -xzf ort.tgz -C ort --strip-components=1 \
 && rm -f ort.tgz /tmp/ort-cache/ort.tgz

# ソースは変更されやすいので、重い手順(上の apt と ORT の取得)より後に COPY する。
# こうするとソースだけ変えたときに、上の層がキャッシュされて再ビルドが速い
WORKDIR /src
COPY CMakeLists.txt .
COPY src ./src
RUN cmake -S . -B build -DORT_DIR=/opt/ort -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" --target infer_server

# ============ 実行用の段階 ============
# CUDA 13 のランタイムと cuDNN 9 が入った公式イメージ。ORT の CUDA プロバイダが必要とする
# libcudart / libcublas / libcurand / libcudnn がここに含まれる(ホストのドライバは Toolkit が注入)
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 AS runtime

# ORT のライブラリ(.so だけ。ヘッダや cmake 設定は実行に要らない)。
# infer_server は RPATH=/opt/ort/lib でビルドされているので、同じ場所に置く
COPY --from=build /opt/ort/lib/*.so* /opt/ort/lib/
COPY --from=build /src/build/infer_server /usr/local/bin/infer_server

# 一般ユーザー。TensorRT のエンジンキャッシュ(trt_cache/)などは作業フォルダに作られる
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
 && mkdir /work && chown app:app /work
USER app
WORKDIR /work

# 8080 = 推論(独自の TCP)、9100 = メトリクス(HTTP の /metrics と /healthz。環境変数 INFER_METRICS_PORT で変更、0 で無効)
EXPOSE 8080 9100
# PID 1 になる。infer_server は SIGTERM のハンドラを持つので docker stop で終了処理(統計の表示)が走る
ENTRYPOINT ["infer_server"]
CMD ["/models/mlp_dyn.onnx", "2", "1", "0", "32", "1", "cuda"]
