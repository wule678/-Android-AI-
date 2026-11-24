// 课程综合实验：本地人脸库（注册 / 识别 / 删除）接口
// 纯内存实现，特征向量在入库存放时保持 L2 归一化，匹配使用余弦相似度。

#ifndef FACESTORE_H
#define FACESTORE_H

#include <string>
#include <vector>

// 计算两个 L2 归一化特征向量的余弦相似度（∈ [-1,1]，越大越像）。
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

// 本地人脸库。用于“经同意的用户注册、重复注册处理、身份识别、未知身份拒识和模板删除”。
class FaceStore
{
public:
    struct Entry
    {
        std::string name;
        std::vector<float> feature;   // L2 归一化特征
    };

    // 注册。若 name 已存在则返回 false（重复注册处理）；否则保存特征并返回 true。
    bool register_face(const std::string& name, const std::vector<float>& feature);

    // 识别。返回最相似且相似度 >= threshold 的名字；若都不达标返回 ""（未知身份拒识）。
    // score 输出最佳余弦相似度。
    std::string recognize(const std::vector<float>& feature, float threshold, float& score) const;

    // 删除指定名字，成功返回 true。
    bool remove(const std::string& name);

    void clear();

    std::vector<std::string> list() const;

    size_t size() const { return entries_.size(); }

private:
    std::vector<Entry> entries_;
};

#endif // FACESTORE_H
