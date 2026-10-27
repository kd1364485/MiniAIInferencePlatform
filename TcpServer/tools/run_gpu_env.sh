#!/bin/bash
# GPU 版の infer_server を動かすための環境を整えて、渡されたコマンドを実行する。
#
# 使い方:
#   tools/run_gpu_env.sh ./build/infer_server models/mlp_dyn.onnx 2 1 0 32 2 cuda
#
# なぜ必要か: ONNX Runtime の CUDA / TensorRT プロバイダは、CUDA 13・cuDNN・TensorRT の共有ライブラリを
#   起動時に探す。それらは Python の pip パッケージ(nvidia-*, tensorrt-cu13)として入れてあるので、
#   場所(site-packages の下)を LD_LIBRARY_PATH に足す。LD_LIBRARY_PATH はプロセスの起動時にしか
#   読まれないので、サーバーを起動する「前」に設定する必要がある(M1 の gpu_bench.py と同じ理由)。
#
# PYTHON 環境変数: 上のパッケージが入った Python(既定: python3)。例: PYTHON=~/OnnxLearn/venv/bin/python
PYTHON=${PYTHON:-python3}
SP=$("$PYTHON" -I -c 'import site; print(site.getsitepackages()[0])') || exit 1

for rel in nvidia/cu13/lib nvidia/cudnn/lib tensorrt_libs; do
  [ -d "$SP/$rel" ] && LD_LIBRARY_PATH="$SP/$rel:$LD_LIBRARY_PATH"
done
[ -d /usr/lib/wsl/lib ] && LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"   # WSL の libcuda.so.1
export LD_LIBRARY_PATH
exec "$@"
