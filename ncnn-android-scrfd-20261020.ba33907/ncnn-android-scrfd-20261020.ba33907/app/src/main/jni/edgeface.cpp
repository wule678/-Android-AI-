// 课程综合实验：人脸识别模块（EdgeFace-XXS）实现
// 结构与预处理流程参考 ncnn-android-scrfd 的 scrfd.cpp。

#include "edgeface.h"

#include <android/asset_manager.h>
#include <android/log.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cpu.h"

// ---------------------------------------------------------------------------
// 模型相关的常量。以下取值以 EdgeFace-XXS 官方 ONNX 导出为准，转换后需核对：
//   - 输入张量名：通常为 "input.1"（参考 scrfd）或 "images"，按实际 param 修改
//   - 输出张量名：EdgeFace 官方最后一层常命名为 "683"，按实际 param 修改
// ---------------------------------------------------------------------------
static const char* EDGEFACE_INPUT_NAME  = "in0";
static const char* EDGEFACE_OUTPUT_NAME = "out0";

// insightface 约定的 112x112 规范脸 5 点坐标
// 顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
static const float CANONICAL_LANDMARK[5][2] = {
    { 38.2946f, 51.6963f },   // 左眼
    { 73.5318f, 51.5014f },   // 右眼
    { 56.0252f, 71.7366f },   // 鼻尖
    { 41.5493f, 92.3655f },   // 左嘴角
    { 70.7299f, 92.2041f }    // 右嘴角
};

EdgeFace::EdgeFace()
    : input_size(112), feature_dim_(512), loaded(false)
{
    // EdgeFace 预处理：(x - 127.5) / 127.5，等价于把像素缩放到 [0,1] 后 (x-0.5)/0.5
    mean_vals[0] = 127.5f;
    mean_vals[1] = 127.5f;
    mean_vals[2] = 127.5f;
    norm_vals[0] = 1.f / 127.5f;
    norm_vals[1] = 1.f / 127.5f;
    norm_vals[2] = 1.f / 127.5f;
}

int EdgeFace::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
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

    // 模型文件尚未放入 assets 时会失败并打印日志，此时 loaded 保持 false，
    // 上层流水线会跳过识别，保证 App 仍可运行（只做检测）。
    int r1 = net.load_param(mgr, parampath);
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFace", "load_param %s ret=%d", parampath, r1);
    if (r1 != 0)
        return -1;
    int r2 = net.load_model(mgr, modelpath);
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFace", "load_model %s ret=%d", modelpath, r2);
    if (r2 != 0)
        return -1;

    loaded = true;
    return 0;
}

int EdgeFace::align_face(const cv::Mat& rgb, const cv::Point2f landmark[5], cv::Mat& aligned) const
{
    // 构造源点（检测关键点）与目标点（规范脸关键点）
    std::vector<cv::Point2f> src(5), dst(5);
    for (int i = 0; i < 5; i++)
    {
        src[i] = landmark[i];
        dst[i] = cv::Point2f(CANONICAL_LANDMARK[i][0], CANONICAL_LANDMARK[i][1]);
    }

    // 最小二乘相似变换：x' = a*x - b*y + tx; y' = b*x + a*y + ty
    // 参考 insightface 的 5 点对齐（skimage.transform.SimilarityTransform 的解析解）。
    float src_mx = 0.f, src_my = 0.f;
    float dst_mx = 0.f, dst_my = 0.f;
    for (int i = 0; i < 5; i++)
    {
        src_mx += src[i].x; src_my += src[i].y;
        dst_mx += dst[i].x; dst_my += dst[i].y;
    }
    src_mx /= 5.f; src_my /= 5.f;
    dst_mx /= 5.f; dst_my /= 5.f;

    // 一次遍历同时累积 denom、a、b：
    //   denom = sum(sx^2 + sy^2)，a = sum(sx*dx + sy*dy)，b = sum(sx*dy - sy*dx)
    float denom = 0.f;
    float a = 0.f, b = 0.f;
    for (int i = 0; i < 5; i++)
    {
        float sx = src[i].x - src_mx;
        float sy = src[i].y - src_my;
        float dx = dst[i].x - dst_mx;
        float dy = dst[i].y - dst_my;
        denom += sx * sx + sy * sy;
        a += sx * dx + sy * dy;
        b += sx * dy - sy * dx;
    }

    if (denom < 1e-6f)
    {
        // 关键点退化，退化为恒等（不缩放不旋转）
        a = 1.f;
        b = 0.f;
    }
    else
    {
        a /= denom;
        b /= denom;
    }

    float tx = dst_mx - (a * src_mx - b * src_my);
    float ty = dst_my - (b * src_mx + a * src_my);

    cv::Mat M(2, 3, CV_32F);
    float* m = M.ptr<float>(0);
    m[0] = a;  m[1] = -b; m[2] = tx;
    m[3] = b;  m[4] = a;  m[5] = ty;
    cv::warpAffine(rgb, aligned, M, cv::Size(input_size, input_size), cv::INTER_LINEAR);

    return 0;
}

int EdgeFace::extract_feature(const cv::Mat& aligned, std::vector<float>& feature)
{
    if (!loaded)
        return -1;

    // 缩放到模型输入尺寸（通常与对齐尺寸一致，112x112）
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        aligned.data, ncnn::Mat::PIXEL_RGB, aligned.cols, aligned.rows,
        input_size, input_size);

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input(EDGEFACE_INPUT_NAME, in);

    ncnn::Mat out;
    ex.extract(EDGEFACE_OUTPUT_NAME, out);

    // 输出为 [1, dim]（或 [dim, 1]，按实际模型调整），展开为向量
    const int dim = out.w * out.h * out.c;
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFace", "extract_feature out.w=%d h=%d c=%d elempack=%d dim=%d", out.w, out.h, out.c, out.elempack, dim);
    feature.resize(dim);
    float norm = 0.f;
    for (int i = 0; i < dim; i++)
    {
        feature[i] = out[i];
        norm += feature[i] * feature[i];
    }
    norm = sqrtf(norm);
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFace", "extract_feature norm=%f first=%f,%f,%f", norm, dim > 0 ? feature[0] : 0.f, dim > 1 ? feature[1] : 0.f, dim > 2 ? feature[2] : 0.f);
    if (norm < 1e-6f)
        return -1;

    // L2 归一化，便于后续余弦相似度匹配
    for (int i = 0; i < dim; i++)
        feature[i] /= norm;

    feature_dim_ = dim;
    return 0;
}
