# PPT 大纲与答辩准备（严格按《综合实验要求》整理）

> 对应作业要求的 PPT 九点内容、现场演示、操作系统指释、个人贡献四块。

---

## 一、PPT 大纲（18 页，严格对应要求的 9 点）

### 第 1 页 · 封面
- 项目名：《基于 Android 平台的轻量化 AI 模型部署与优化 —— 前置摄像头人脸检测、识别和活体检测》
- 成员姓名 + 学号 + 分工（A：AI 模型侧，B：操作系统/系统侧）

### 第 2 页 · 选题背景（对应要求第 1 点）
- 操作系统课程核心：进程/线程/IPC/信号量/I/O 事件
- AI 从算法到真实设备的约束：算力、内存、功耗、延迟
- 为什么选 Android 真机 + 离线端侧推理

### 第 3 页 · 实验目标与验收任务（对应要求第 1 点）
- 三项 AI 功能（检测/识别/活体，必须真实端侧模型）
- 五类 OS 机制（进程/IPC/线程/信号量/I/O 事件）
- 验收任务：可安装 APK + 源工程 + README + Git + PPT + 真机演示

### 第 4 页 · 系统架构（对应要求第 2 点）
- 双进程架构图：主进程（摄像头+检测+预处理）+ :inference 进程（识别+活体）
- 三模型数据流：检测 → 对齐 → 识别 / 活体

### 第 5 页 · 三模型数据流与线程/进程分工（对应要求第 2 点）
- 主进程：摄像头线程、消费者线程（Semaphore 有界队列）
- :inference 进程：AIDL Service、推理线程
- 数据流图 + 各线程/进程职责表

### 第 6 页 · 人脸检测模型 SCRFD-500M-KPS（对应要求第 3 点）
- 模型简介、输入输出、预处理
- Android 部署：ncnn + opencv，NDK Camera 采集

### 第 7 页 · 人脸识别模型（对应要求第 3 点）
- MobileFaceNet（insightface w600k_mbf，512 维特征）
- 关键点对齐（相似变换）+ 余弦相似度匹配 + 注册/拒识

### 第 8 页 · 活体检测模型 MiniFASNetV2（对应要求第 3 点）
- 3 类输出、真人/攻击分类、连续帧 EMA 稳定
- 部署细节：模型转换（pnnx → ncnn）、预处理

### 第 9 页 · 进程管理与 Binder IPC（对应要求第 4 点）
- 双进程：`com.tencent.scrfdncnn` + `:inference`
- AIDL 接口（IInferenceService）+ Service 绑定 + PID 展示
- IPC 请求统计（次数/时延）

### 第 10 页 · 单进程 vs 双进程比较（对应要求第 4 点）
- 前后对比数据：FPS +35%（20→27）、检测耗时 -71%（51→15ms）
- 结论：双进程把耗时推理移出主进程，UI/采集更流畅

### 第 11 页 · 线程调度（对应要求第 5 点）
- 线程分工：UI 主线程、摄像头回调线程、消费者线程、推理线程
- 线程数 1/2/4 对比实验

### 第 12 页 · 信号量生产者-消费者（对应要求第 5 点）
- Semaphore 实现（计数信号量）
- 摄像头=生产者、推理=消费者、有界队列 + 丢帧
- 队列深度 1/3/5 对比实验（深度 3 最优）

### 第 13 页 · 摄像头 I/O 事件链（对应要求第 6 点）
- 分层图：硬件→HAL→Camera2→应用回调→检测→队列→推理→UI
- 时间戳：det / wait / prep（各阶段耗时）
- 说明：系统 Trace/Perfetto 为可选加分项，已做应用层时间戳

### 第 14 页 · 优化前后指标（对应要求第 7 点）
- IPC 改造前后：FPS、检测耗时对比
- 模型 fp16（体积/耗时）

### 第 15 页 · 统计图与精度代价（对应要求第 7 点）
- 各阶段耗时统计图（均值/P50/P95）
- 识别准确率 73%、活体对翻拍失效的精度代价

### 第 16 页 · 典型故障与局限性（对应要求第 8 点）
- EdgeFace（EdgeNeXt 结构）pnnx 转换失败 → 换 MobileFaceNet
- 活体对屏幕翻拍失效（0%）的原因分析
- 识别受光照/角度影响

### 第 17 页 · 创新性贡献（对应要求第 9 点）
- 完整 IPC 数据流（识别/活体真正走 AIDL）
- Semaphore 有界队列 + 丢帧控制
- I/O 事件链时间戳

### 第 18 页 · 成员分工 + Git 协作证据 + 总结（对应要求第 9 点）
- A/B 分工表
- Git commit 历史截图
- 总结 + 展望

---

## 二、现场演示脚本（8-10 分钟）

> 按答辩要求：真机演示 + 操作系统指释。

### 演示 1：打开 App（30 秒）
1. 真机打开 App，展示前置摄像头画面 + 人脸检测框 + 5 关键点
2. 指一下：画面左上角的 FPS、det/wait/prep 耗时、queue/dropped/done 状态
3. 说：这是实时状态页，能看到检测、队列、丢帧等系统状态

### 演示 2：身份注册与匹配（1 分钟）
1. 点「注册人脸」→ 输入名字 → 确定
2. 脸对准摄像头，展示「识别: 名字」
3. 换一个人（没注册的），展示「识别: 未知」（拒识）

### 演示 3：活体检测（1 分钟）
1. 真人：展示「活体: 真人」
2. 打印照片：展示「活体: 攻击」
3. （说明：屏幕翻拍会被误判，这是模型局限，后面讲）

### 演示 4：操作系统指释（1 分钟，答辩关键！）
边演示边指出：
1. **进程**：切到系统，展示 `com.tencent.scrfdncnn` 和 `com.tencent.scrfdncnn:inference` 两个进程（或指界面上的 PID 显示）
2. **线程**：说明消费者线程在哪个进程、推理线程在哪个进程
3. **IPC 接口**：指 AIDL 的 `IInferenceService`（extractFeature/detectLiveness/recognizeFace）
4. **Semaphore**：指画面上的 queue/dropped，说明信号量在 `scrfdncnn.cpp` 的 `g_sem_space.try_acquire()/release()`
5. **I/O 回调 vs 中断**：解释摄像头是"回调"（HAL 通过回调把帧传上来），不是"中断"（CPU 级的中断由驱动处理）

---

## 三、答辩准备（可能被问的问题 + 答案）

### 1. 进程和线程实际创建位置在哪？
- 进程：`AndroidManifest.xml` 里 `<service android:process=":inference">` 声明了独立进程；主进程是默认进程。
- 线程：消费者线程在 `scrfdncnn.cpp` 的 `JNI_OnLoad` 里 `std::thread(consumer_loop)` 创建；推理线程由 Service 进程里的 ncnn 内部线程池管理。

### 2. IPC 接口是什么？
- AIDL 接口 `IInferenceService`，方法：`extractFeature`（提特征）、`detectLiveness`（活体）、`recognizeFace`（识别）、`registerFace/deleteFace`（人脸库）。
- 主进程通过 `bindService` + `IInferenceService.Stub.asInterface` 拿到 Binder 代理。

### 3. Semaphore 在哪获取和释放？
- 获取：`scrfdncnn.cpp` 生产者（on_image_render）里 `g_sem_space.try_acquire()`，队列满则丢帧。
- 释放：消费者线程 `consumer_loop` 取出任务后 `g_sem_space.release()`。

### 4. 摄像头 I/O 回调与中断的区别？
- 回调：应用层通过 Camera2/NDK Camera 注册回调，HAL 层每来一帧通过回调把帧数据传给应用（软件层的事件通知）。
- 中断：CPU 硬件级的中断（如摄像头 DMA 完成中断），由内核驱动处理，应用层感知不到。
- 应用关心的是"回调"，中断在驱动层。

### 5. 为什么用 MobileFaceNet 不用 EdgeFace？
- 作业推荐 EdgeFace-XXS，但它的骨干 EdgeNeXt 用了多维注意力（reshape/permute/unbind），pnnx 转换这些算子有问题（unbind 生成 ncnn 不认识的层、推理崩溃）。
- 换成了 MobileFaceNet（纯卷积 + PReLU），pnnx 转换完美，作业也允许"MobileFaceNet、轻量 ArcFace 主干"。

### 6. 屏幕翻拍为什么被误判为真人？
- MiniFASNetV2 是 RGB 静默活体，靠频域/纹理特征区分真假。
- 屏幕翻拍本身发光、对比度高，频域特征与真人接近，模型难以区分。
- 这是 RGB 静默活体的已知局限，报告中如实说明。

---

## 四、个人贡献分工（答辩时各答各的）

| 成员 | 负责部分 | 需回答的关键技术问题 |
|---|---|---|
| A（模型侧） | 检测/识别/活体模型、模型转换、三模型流水线 | 模型结构、预处理、对齐、余弦匹配、为什么换 MobileFaceNet |
| B（系统侧） | 双进程 IPC、Semaphore、线程、I/O 事件链、状态页 | 进程/线程创建位置、AIDL 接口、Semaphore 获取/释放、回调 vs 中断 |
