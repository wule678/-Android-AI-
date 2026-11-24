// IInferenceService.aidl
package com.tencent.scrfdncnn;

// AI 推理服务接口（运行在独立进程 :inference，通过 Binder 通信）
interface IInferenceService {
    // 加载识别 + 活体模型
    boolean loadModels();

    // 推理服务进程 PID（用于展示进程角色）
    int getPid();

    // 从已对齐的 RGB 图提取人脸特征（L2 归一化向量，对齐由主进程完成）
    float[] extractFeature(in byte[] aligned, int width, int height);

    // 活体检测，返回真人分数 [0,1]
    float detectLiveness(in byte[] rgb, int width, int height);

    // 注册 / 识别 / 删除
    boolean registerFace(String name, in float[] feature);
    String recognizeFace(in float[] feature, float threshold);
    boolean deleteFace(String name);
    void clearFaces();

    // IPC 统计
    long getRequestCount();
    long getTotalLatencyMs();
}
