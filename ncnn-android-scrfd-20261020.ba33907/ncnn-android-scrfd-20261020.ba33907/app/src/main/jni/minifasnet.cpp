// 课程综合实验：活体检测模块（MiniFASNetV2）实现
// 结构与预处理流程参考 ncnn-android-scrfd 的 scrfd.cpp。

#include "minifasnet.h"

#include <android/asset_manager.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cpu.h"

// ---------------------------------------------------------------------------
// 模型相关的常量。以下取值以 MiniFASNetV2 官方 ONNX 导出为准，转换后需核对：
//   - 输入张量名 / 输出张量名：按实际 param 修改
//   - 输出语义：部分导出版本输出二分类 logits，部分直接输出 sigmoid 后的分数，
//     需确认哪个类别对应“真人”，必要时在 detect_liveness 里取反。
// ---------------------------------------------------------------------------
static const char* MINIFASNET_INPUT_NAME  = "in0";
static const char* MINIFASNET_OUTPUT_NAME = "out0";

MiniFASNet::MiniFASNet()
    : input_size(80), threshold(0.5f), loaded(false)
{
    // 官方 MiniFASNetV2 预处理：BGR 输入，值域 [0,255]，不做归一化。
    // （官方 test.py 用 cv2.imread 得到 BGR，ToTensor 只做 HWC→CHW，不 div(255)、不减 mean）
    // 所以 mean=0, norm=1，保持 [0,255]。
    mean_vals[0] = 0.f;
    mean_vals[1] = 0.f;
    mean_vals[2] = 0.f;
    norm_vals[0] = 1.f;
    norm_vals[1] = 1.f;
    norm_vals[2] = 1.f;
}

int MiniFASNet::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    loaded = false;
    net.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = use_gpu;
#endif

    net.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    if (net.load_param(mgr, parampath) != 0)
        return -1;
    if (net.load_model(mgr, modelpath) != 0)
        return -1;

    loaded = true;
    return 0;
}

int MiniFASNet::detect_liveness(const cv::Mat& face_crop, float& real_score) const
{
    if (!loaded)
        return -1;

    // 官方模型吃 BGR（cv2.imread 是 BGR）。这里 face_crop 是 RGB，用 PIXEL_BGR 读取等效于 R/B 交换得到 BGR
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        face_crop.data, ncnn::Mat::PIXEL_BGR, face_crop.cols, face_crop.rows,
        input_size, input_size);

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input(MINIFASNET_INPUT_NAME, in);

    ncnn::Mat out;
    ex.extract(MINIFASNET_OUTPUT_NAME, out);

    // 输出为 3 类 logits（num_classes=3）。官方 test.py 用 argmax，index 1 = 真人。
    // 这里对 3 类做 softmax，取第 1 类作为“真人分数” real_score。
    // TODO: 转换后核对输出形状与类别顺序（若与官方不一致需调整）。
    const int n = out.w * out.h * out.c;
    if (n < 3)
        return -1;

    float max_logit = out[0];
    if (out[1] > max_logit) max_logit = out[1];
    if (out[2] > max_logit) max_logit = out[2];

    float exp0 = expf(out[0] - max_logit);
    float exp1 = expf(out[1] - max_logit);
    float exp2 = expf(out[2] - max_logit);
    float sum = exp0 + exp1 + exp2;

    real_score = exp1 / sum;

    return 0;
}
