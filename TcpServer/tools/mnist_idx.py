"""MNIST の IDX ファイルを numpy に読む小さな共通モジュール(学習・INT8 測定の両方で使う)。

IDX のヘッダは「ビッグエンディアン」(上位バイトが先)。x86 はリトルエンディアンなので、
struct の '>I' で読む。画像ファイル: magic(4B) 枚数(4B) 行(4B) 列(4B) + 画素。ラベルファイル: magic(4B) 枚数(4B) + ラベル。
(C++ の infer_bench.cpp / OnnxLearn/infer.cpp の read_be32 と同じ仕事を Python でやっている)
"""
import os
import struct

import numpy as np


def load_images(raw_dir, name):
    """name 例: 't10k-images-idx3-ubyte'。戻り値: uint8 の (N, 1, 28, 28)。"""
    with open(os.path.join(raw_dir, name), "rb") as f:
        data = f.read()
    n = struct.unpack(">I", data[4:8])[0]
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, 1, 28, 28)


def load_labels(raw_dir, name):
    """name 例: 't10k-labels-idx1-ubyte'。戻り値: uint8 の (N,)。"""
    with open(os.path.join(raw_dir, name), "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def preprocess(u8):
    """uint8 → float32 の -1..1。infer_server.cpp / 学習時の Normalize((0.5,),(0.5,)) と同じ。"""
    return u8.astype(np.float32) / 255.0 * 2.0 - 1.0
