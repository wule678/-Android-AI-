// 课程综合实验：本地人脸库（注册 / 识别 / 删除）实现

#include "facestore.h"

#include <math.h>

#include <algorithm>

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size())
        return 0.f;

    float dot = 0.f;
    for (size_t i = 0; i < a.size(); i++)
        dot += a[i] * b[i];

    // 特征已 L2 归一化，点积即余弦相似度；这里再兜底防呆
    return dot;
}

bool FaceStore::register_face(const std::string& name, const std::vector<float>& feature)
{
    // 重复注册处理：同名已存在则拒绝
    for (size_t i = 0; i < entries_.size(); i++)
    {
        if (entries_[i].name == name)
            return false;
    }

    Entry e;
    e.name = name;
    e.feature = feature;
    entries_.push_back(e);
    return true;
}

std::string FaceStore::recognize(const std::vector<float>& feature, float threshold, float& score) const
{
    score = -1.f;
    std::string best_name;

    for (size_t i = 0; i < entries_.size(); i++)
    {
        float s = cosine_similarity(feature, entries_[i].feature);
        if (s > score)
        {
            score = s;
            best_name = entries_[i].name;
        }
    }

    if (score < threshold)
    {
        score = best_name.empty() ? 0.f : score;
        return "";   // 未知身份拒识
    }

    return best_name;
}

bool FaceStore::remove(const std::string& name)
{
    for (std::vector<Entry>::iterator it = entries_.begin(); it != entries_.end(); ++it)
    {
        if (it->name == name)
        {
            entries_.erase(it);
            return true;
        }
    }
    return false;
}

void FaceStore::clear()
{
    entries_.clear();
}

std::vector<std::string> FaceStore::list() const
{
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (size_t i = 0; i < entries_.size(); i++)
        names.push_back(entries_[i].name);
    return names;
}
