# 基于 Android 的轻量化 AI 模型部署与优化

《操作系统》综合实验 · 前置摄像头人脸检测、识别和活体检测

## 项目简介

本工程是《操作系统》课程综合实验项目，基于 **ncnn + opencv-mobile** 在 Android 真机上部署三个真实的深度学习模型，并落地操作系统的核心机制（进程、线程、IPC、信号量、I/O 事件）。

| 功能 | 模型 | 说明 |
|---|---|---|
| 人脸检测 | SCRFD-500M-KPS | 带 5 关键点 |
| 人脸识别 | MobileFaceNet（insightface `w600k_mbf`） | WebFace600K 训练，512 维特征 |
| 活体检测 | MiniFASNetV2 | 静默活体，反打印照片/屏幕翻拍 |

## 功能特性

1. 前置摄像头实时采集 + 人脸检测（框 + 5 关键点 + 置信度）
2. 人脸注册 / 识别 / 删除 / 清空（余弦相似度本地匹配，未知拒识）
3. 活体检测（真人/攻击分类 + 风险分数 + 连续帧 EMA 稳定）
4. **双进程 IPC**：UI/摄像头进程 + 独立 AI 推理服务进程（Binder/AIDL）
5. **Semaphore 生产者-消费者**：摄像头=生产者、推理=消费者，有界队列 + 丢帧控制
6. **I/O 事件链时间戳**：检测 / 队列等待 / 识别 / 活体 各阶段耗时实时展示

## 系统架构

```
[主进程 com.tencent.scrfdncnn]
  摄像头(NDK Camera) → SCRFD 检测 → 生产者(有界队列, Semaphore)
                                    ↓
  消费者线程: 对齐 → MobileFaceNet 识别 → MiniFASNetV2 活体
                                    ↓
  渲染标注(英文) + TextView(中文结果)

[推理进程 com.tencent.scrfdncnn:inference]
  InferenceService (AIDL): 加载识别/活体模型，提供推理接口
```

## 环境要求

- Android Studio（AGP 8.7.3）
- Android SDK：compileSdk 33 / targetSdk 35 / minSdk 24
- NDK `30.0.16248370`
- Gradle 8.12、JDK 17
- ncnn `ncnn-20261020-android-vulkan`（已内置 `app/src/main/jni/`）
- opencv-mobile `4.14.0`（已内置）

## 编译步骤

1. 用 Android Studio 打开工程，或命令行：
   ```bash
   set JAVA_HOME=<你的 JDK 17 路径>
   gradlew assembleDebug
   ```
2. 等待 CMake 编译 native 代码
3. 产物：`app/build/outputs/apk/debug/com.tencent.scrfdncnn-debug.apk`

> 模型文件（`app/src/main/assets/*.param/*.bin`）已转换并内置，直接构建即可。

## 模型来源与转换

| 模型 | 来源 | 转换方式 |
|---|---|---|
| SCRFD-500M-KPS | ncnn 官方示例 | 已内置（assets） |
| MobileFaceNet | insightface `buffalo_sc` 包内 `w600k_mbf.onnx` | pnnx 转 ncnn |
| MiniFASNetV2 | Silent-Face-Anti-Spoofing | pnnx 转 ncnn（fp16） |

转换工具：pnnx（`pip install pnnx`），详细步骤见 `tools/模型转换说明.md`。

## 运行方式

1. 安装 APK 到 Android 真机（需前置摄像头 + Camera2 HAL3）
2. 首次启动授予相机权限
3. 对准人脸：画面显示检测框 + 关键点 + 识别/活体标签
4. 点「注册人脸」输入名字 → 再对准人脸自动识别
5. 用打印照片 / 屏幕翻拍测试活体（应判为 spoof）

## 已知限制

- 识别模型为 MobileFaceNet（EdgeFace-XXS 的 EdgeNeXt 结构 pnnx 转换有算子问题，故换用 MobileFaceNet）
- 活体检测对打印照片 / 屏幕翻拍的区分度受环境光照影响
- 很老的设备缺少 HAL3 camera 接口可能崩溃
- 画面上的识别标签为英文（OpenCV putText 不支持中文），中文结果显示在界面的 TextView

## 参考

- ncnn：https://github.com/Tencent/ncnn
- opencv-mobile：https://github.com/nihui/opencv-mobile
- insightface：https://github.com/deepinsight/insightface
- Silent-Face-Anti-Spoofing：https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
