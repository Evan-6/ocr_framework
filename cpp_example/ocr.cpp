// pch.cpp: 与预编译标头对应的源文件

#include "vision/ocr.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "config/resource_config.h"
#include "logging/logger.h"
#include "vision/ocr_label_decoder.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <string>
#include <windows.h>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

using namespace cv;




namespace {

// data_1 5MB model interface (see runs/data_1_5mb/model_meta.json):
//   input  "input"  float32 (1, 3, 64, 224), RGB, x = pixel/127.5 - 1.0
//   output "logits" float32 (1, T, num_classes); argmax per step -> CTC decode
//   trained with resize_mode "stretch" -> preprocessing is a single resize.
constexpr int kImgH = 64;
constexpr int kImgW = 224;

Ort::Session* session = nullptr;

Ort::Env& GetOcrEnv()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test");
    return env;
}

void ConfigureOcrSessionOptions(Ort::SessionOptions& session_options)
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

bool TryUseDirectMlProvider(Ort::SessionOptions& session_options)
{
    try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
        logging::Info(u8"ONNX Runtime OCR 使用 DirectMLExecutionProvider");
        return true;
    }
    catch (const std::exception& e) {
        logging::Warn(u8"ONNX Runtime DirectML provider 初始化失敗，OCR fallback CPU: ", e.what());
        return false;
    }
}

} // namespace

void InitOcr() {
    Close();

    HMODULE ghmodule = GetModuleHandle(nullptr);
    HRSRC hrsrc = FindResource(ghmodule, MAKEINTRESOURCE(config::Resources::OcrModelResourceId), config::Resources::ModelResourceType);
    HGLOBAL hg = LoadResource(ghmodule, hrsrc);
    unsigned char* addr = (unsigned char*)(LockResource(hg));
    DWORD size = SizeofResource(ghmodule, hrsrc);
    Ort::SessionOptions session_options;
    ConfigureOcrSessionOptions(session_options);
    const bool using_directml = TryUseDirectMlProvider(session_options);
    try {
        session = new Ort::Session(GetOcrEnv(), addr, size, session_options);
    }
    catch (const std::exception& e) {
        if (!using_directml) {
            throw;
        }

        logging::Warn(u8"ONNX Runtime DirectML session 建立失敗，OCR fallback CPU: ", e.what());
        Ort::SessionOptions cpu_session_options;
        ConfigureOcrSessionOptions(cpu_session_options);
        session = new Ort::Session(GetOcrEnv(), addr, size, cpu_session_options);
    }

}




std::string Classification(cv::Mat& originalImage) {
    if (session == nullptr) {
        logging::Warn(u8"OCR 尚未初始化");
        return {};
    }

    logging::Debug(u8"OCR 圖像原始尺寸: ", originalImage.rows, "x", originalImage.cols);

    // 前處理：模型以 resize_mode "stretch" 訓練，且 data_1 影像尺寸固定，
    // 因此只需直接 resize 到網路輸入大小並轉為 RGB（無模糊/裁切/padding）。
    cv::Mat rgb;
    if (originalImage.channels() == 4) {
        cv::cvtColor(originalImage, rgb, cv::COLOR_BGRA2RGB);
    }
    else if (originalImage.channels() == 1) {
        cv::cvtColor(originalImage, rgb, cv::COLOR_GRAY2RGB);
    }
    else {
        cv::cvtColor(originalImage, rgb, cv::COLOR_BGR2RGB);
    }
    cv::resize(rgb, rgb, cv::Size(kImgW, kImgH), 0, 0, cv::INTER_LINEAR);

    // 轉成 CHW float，normalize x/127.5 - 1.0（與訓練一致）
    std::vector<float> input_tensor_values(static_cast<size_t>(1) * 3 * kImgH * kImgW);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < kImgH; ++h) {
            for (int w = 0; w < kImgW; ++w) {
                const float pixel = rgb.at<cv::Vec3b>(h, w)[c];
                input_tensor_values[(static_cast<size_t>(c) * kImgH + h) * kImgW + w] =
                    pixel / 127.5f - 1.0f;
            }
        }
    }

    std::vector<const char*> input_node_names_real = { "input" };
    std::vector<const char*> output_node_names_real = { "logits" };
    std::vector<int64_t> input_node_dims{ 1, 3, kImgH, kImgW }; // 1为批次，3为通道数
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // 运行模型
    auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_node_names_real.data(), &input_tensor, 1, output_node_names_real.data(), 1);
    if (output_tensors.size() != 1 || !output_tensors.front().IsTensor()) {
        logging::Warn(u8"OCR 模型輸出不是單一 tensor");
        return {};
    }

    // 輸出為 float logits (1, T, num_classes)；每個時間步取 argmax 得到 label 序列，
    // 再交給既有的 DecodeLabelSequence 做 CTC 合併（去重 + 去 blank）。
    const Ort::TensorTypeAndShapeInfo output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        logging::Warn(u8"OCR 模型輸出型別不是 float");
        return {};
    }

    const std::vector<int64_t> output_shape = output_info.GetShape(); // {1, T, num_classes}
    if (output_shape.size() != 3) {
        logging::Warn(u8"OCR 模型輸出維度不是 3");
        return {};
    }
    const int64_t timesteps = output_shape[1];
    const int64_t num_classes = output_shape[2];
    const float* logits = output_tensors.front().GetTensorData<float>();

    std::vector<int64_t> labels(static_cast<size_t>(timesteps));
    for (int64_t t = 0; t < timesteps; ++t) {
        const float* row = logits + t * num_classes;
        labels[t] = std::max_element(row, row + num_classes) - row;
    }

    const vision::ocr::LabelDecodeResult decoded = vision::ocr::DecodeLabelSequence(labels);
    if (decoded.had_invalid_label) {
        logging::Warn(u8"OCR 模型輸出包含未知 label: ", decoded.first_invalid_label);
    }
    return decoded.text;
}

void Close() {
    delete session;
    session = nullptr;
}
