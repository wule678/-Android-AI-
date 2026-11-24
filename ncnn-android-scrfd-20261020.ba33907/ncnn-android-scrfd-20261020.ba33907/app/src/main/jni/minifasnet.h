// 课程综合实验：活体检测模块（MiniFASNetV2）接口
// 参考 ncnn-android-scrfd 的 scrfd.h 结构组织，模型推理使用 ncnn。

#ifndef MINIFASNET_H
#define MINIFASNET_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct AAssetManager;

// MiniFASNetV2 人脸反欺骗（PAD）模型封装。
// 输入：人脸裁剪图（内部会缩放到 80x80）；输出：真人概率 real_score ∈ [0,1]。
class MiniFASNet
{
public:
    MiniFASNet();

    // 从 assets 加载模型。modeltype 对应 assets 里的文件名前缀
    // （例如 "minifasnetv2" -> minifasnetv2.param / minifasnetv2.bin）。
    int load(AAssetManager* mgr, const char* modeltype = "minifasnetv2", bool use_gpu = false);

    bool is_loaded() const { return loaded; }

    // 对一张人脸裁剪图做活体判定。
    // real_score 越接近 1 越像真人，越接近 0 越像攻击（打印照片/屏幕翻拍）。
    int detect_liveness(const cv::Mat& face_crop, float& real_score) const;

    // 判定阈值：real_score >= threshold 判为真人，否则判为攻击。
    // 默认 0.5，需在真实数据上标定。
    float threshold;

private:
    ncnn::Net net;
    int input_size;      // MiniFASNetV2 输入尺寸，通常 80
    float mean_vals[3];
    float norm_vals[3];
    bool loaded;
};

#endif // MINIFASNET_H
