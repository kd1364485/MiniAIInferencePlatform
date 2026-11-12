// M4: mnist.onnx を C++ の ONNX Runtime で推論する
// Pythonの infer_mnist.py と同じ仕事をC++でやる
// MNISTのIDXバイナリを直接 ifstream で読む(Phase 1流のファイルパース)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

// IDXファイルのヘッダはビッグエンディアン。リトルエンディアンのx86で読むため変換
static uint32_t read_be32(std::ifstream& f) {
    uint8_t b[4];
    f.read((char*)b, 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

int main() {
    // ========== 1. ONNX Runtimeセッションを作る ==========
    // Pythonの ort.InferenceSession(...) に相当
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");
    Ort::Session session(env, "mnist.onnx", Ort::SessionOptions{});

    // ========== 2. MNISTテスト画像を直接パースする ==========
    // IDXフォーマット: magic(4B) count(4B) rows(4B) cols(4B) + 画素データ
    std::ifstream img_f("data/MNIST/raw/t10k-images-idx3-ubyte", std::ios::binary);
    std::ifstream lbl_f("data/MNIST/raw/t10k-labels-idx1-ubyte", std::ios::binary);
    if (!img_f || !lbl_f) { std::cerr << "data open failed\n"; return 1; }

    read_be32(img_f);                        // magic (0x803) 読み飛ばし
    uint32_t n_img = read_be32(img_f);       // 画像数
    uint32_t rows  = read_be32(img_f);       // 28
    uint32_t cols  = read_be32(img_f);       // 28
    read_be32(lbl_f);                        // magic (0x801)
    uint32_t n_lbl = read_be32(lbl_f);

    std::vector<uint8_t> pixels(rows * cols);
    std::vector<float>   input(784);         // 正規化後のfloat入力

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {1, 1, 28, 28};       // (batch, ch, H, W)
    const char* in_names[]  = {"input"};
    const char* out_names[] = {"output"};

    std::cout << "idx  pred  true\n";
    int correct = 0;
    for (uint32_t i = 0; i < 10 && i < n_img; ++i) {
        img_f.read((char*)pixels.data(), pixels.size());
        uint8_t label;
        lbl_f.read((char*)&label, 1);

        // 前処理: pixel/255 → (v-0.5)/0.5  (trainと同じNormalize)
        for (int j = 0; j < 784; ++j)
            input[j] = pixels[j] / 255.0f * 2.0f - 1.0f;

        // ========== 3. 推論 ==========
        // inputテンソルを作って sess.run 相当を呼ぶ
        Ort::Value in = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape, 4);
        auto outs = session.Run(Ort::RunOptions{},
                                in_names, &in, 1, out_names, 1);
        float* logits = outs[0].GetTensorMutableData<float>();  // 10個のスコア

        int pred = std::max_element(logits, logits + 10) - logits;  // argmax
        std::cout << i << "    " << pred << "     " << (int)label
                  << (pred == label ? "  OK" : "  NG") << "\n";
        correct += (pred == label);
    }
    std::cout << correct << "/10 correct\n";
}
