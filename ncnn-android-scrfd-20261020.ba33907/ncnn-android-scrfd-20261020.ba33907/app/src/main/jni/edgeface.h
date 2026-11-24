// 课程综合实验：人脸识别模块（EdgeFace-XXS）接口
// 参考 ncnn-android-scrfd 的 scrfd.h 结构组织，模型推理使用 ncnn。

#ifndef EDGEFACE_H
#define EDGEFACE_H

#include <opencv2/core/core.hpp>

#include <net.h>

#include <string>
#include <vector>

// 前置声明，避免在头文件里引入 android/asset_manager.h
struct AAssetManager;

// EdgeFace-XXS 人脸识别模型封装。
// 输入：对齐后的 112x112 RGB 人脸；输出：L2 归一化的 512 维特征向量。
class EdgeFace
{
public:
    EdgeFace();

    // 从 assets 加载模型。modeltype 对应 assets 里的文件名前缀
    // （例如 "edgeface_xxs" -> edgeface_xxs.param / edgeface_xxs.bin）。
    // 返回 0 表示成功，非 0 表示失败（例如模型文件尚未放入 assets）。
    int load(AAssetManager* mgr, const char* modeltype = "edgeface_xxs", bool use_gpu = false);

    bool is_loaded() const { return loaded; }

    // 用 SCRFD 的 5 个关键点做相似变换，把人脸对齐到 112x112 的规范脸。
    // landmark 顺序（与 insightface / SCRFD 一致，需在真机上核对）：
    //   [0]=左眼  [1]=右眼  [2]=鼻尖  [3]=左嘴角  [4]=右嘴角
    int align_face(const cv::Mat& rgb, const cv::Point2f landmark[5], cv::Mat& aligned) const;

    // 从对齐人脸提取特征向量（已做 L2 归一化）。
    int extract_feature(const cv::Mat& aligned, std::vector<float>& feature);

    int feature_dim() const { return feature_dim_; }

private:
    ncnn::Net net;
    int input_size;      // EdgeFace-XXS 输入尺寸，通常 112
    int feature_dim_;    // 输出特征维度，EdgeFace 通常 512
    float mean_vals[3];
    float norm_vals[3];
    bool loaded;
};

#endif // EDGEFACE_H
