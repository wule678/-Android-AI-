// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include <platform.h>
#include <benchmark.h>

#include "scrfd.h"
#include "edgeface.h"
#include "minifasnet.h"
#include "facestore.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static void draw_label(cv::Mat& rgb, const std::string& text, int y, const cv::Scalar& fg = cv::Scalar(0, 0, 0))
{
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = 0;
    if (x + label_size.width > rgb.cols)
        x = rgb.cols - label_size.width;
    if (x < 0) x = 0;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, fg, 1);
}

static SCRFD* g_scrfd = 0;
static ncnn::Mutex lock;

static EdgeFace* g_edgeface = 0;
static MiniFASNet* g_minifasnet = 0;
static FaceStore g_facestore;

// 识别/活体流水线共享状态（由 lock 保护）
static std::vector<float> g_last_feature;
static std::string g_last_name;
static float g_last_similarity = 0.f;
static float g_last_liveness = 0.f;
static float g_recog_threshold = 0.5f;
static float g_liveness_threshold = 0.5f;

// ---------------------------------------------------------------------------
// 计数信号量（用 mutex + condition_variable 实现）
// 作业要求：摄像头=生产者、模型推理=消费者，有界在途任务控制。
// 许可数即队列空位；异常/超时路径也能通过 try_acquire/release 释放许可。
// ---------------------------------------------------------------------------
class Semaphore
{
public:
    explicit Semaphore(int n) : count_(n) {}

    void acquire()
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return count_ > 0; });
        count_--;
    }

    // 非阻塞获取许可：成功返回 true，失败（无空位）返回 false —— 用于丢帧控制
    bool try_acquire()
    {
        std::unique_lock<std::mutex> lk(m_);
        if (count_ > 0)
        {
            count_--;
            return true;
        }
        return false;
    }

    void release()
    {
        {
            std::unique_lock<std::mutex> lk(m_);
            count_++;
        }
        cv_.notify_one();
    }

    int available() const
    {
        std::unique_lock<std::mutex> lk(m_);
        return count_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    int count_;
};

// 生产者-消费者：摄像头=生产者，识别/活体=消费者
struct FaceTask
{
    cv::Mat rgb;              // 帧拷贝（用于对齐 + 活体裁剪）
    cv::Point2f landmark[5];  // 5 关键点
    cv::Rect_<float> rect;    // 人脸框
    double enqueue_time;      // 入队时间戳（ms），用于计算队列等待耗时
};

static const int g_max_queue = 3;              // 有界队列深度（= 信号量初始许可数）
static std::queue<FaceTask> g_task_queue;
static std::mutex g_queue_mutex;
static std::condition_variable g_queue_cv;
static Semaphore g_sem_space(g_max_queue);     // 空位信号量
static std::atomic<int> g_dropped_frames(0);   // 丢帧数
static std::atomic<int> g_processed_frames(0); // 已处理帧数
static std::atomic<bool> g_consumer_running(false);
// I/O 事件链各阶段耗时（毫秒）
static std::atomic<long> g_detect_ms(0);       // 检测耗时
static std::atomic<long> g_queue_wait_ms(0);   // 队列等待（入队→出队）
static std::atomic<long> g_recog_ms(0);        // 识别耗时
static std::atomic<long> g_liveness_ms(0);     // 活体耗时
static std::thread g_consumer_thread;

// 消费者：对一帧做人脸对齐 + 特征提取 + 识别 + 活体
static void process_task(const FaceTask& task)
{
    ncnn::MutexLockGuard g(lock);

    // 人脸对齐 + 特征提取 + 识别
    if (g_edgeface && g_edgeface->is_loaded())
    {
        double t0 = ncnn::get_current_time();
        cv::Mat aligned;
        if (g_edgeface->align_face(task.rgb, task.landmark, aligned) == 0)
        {
            std::vector<float> feature;
            if (g_edgeface->extract_feature(aligned, feature) == 0)
            {
                g_last_feature = feature;
                float score = 0.f;
                std::string name = g_facestore.recognize(feature, g_recog_threshold, score);
                g_last_name = name;
                g_last_similarity = score;
            }
        }
        g_recog_ms = (long)(ncnn::get_current_time() - t0);
    }

    // 活体检测 + 连续帧稳定（EMA 平滑）
    if (g_minifasnet && g_minifasnet->is_loaded())
    {
        double t0 = ncnn::get_current_time();
        const float scale = 2.7f;
        float cx = task.rect.x + task.rect.width * 0.5f;
        float cy = task.rect.y + task.rect.height * 0.5f;
        float w = task.rect.width * scale;
        float h = task.rect.height * scale;
        int x0 = (int)(cx - w * 0.5f); if (x0 < 0) x0 = 0;
        int y0 = (int)(cy - h * 0.5f); if (y0 < 0) y0 = 0;
        int x1 = (int)(cx + w * 0.5f); if (x1 > task.rgb.cols) x1 = task.rgb.cols;
        int y1 = (int)(cy + h * 0.5f); if (y1 > task.rgb.rows) y1 = task.rgb.rows;
        if (x1 > x0 && y1 > y0)
        {
            cv::Mat face_crop = task.rgb(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
            float real_score = 0.f;
            if (g_minifasnet->detect_liveness(face_crop, real_score) == 0)
            {
                if (g_last_liveness == 0.f)
                    g_last_liveness = real_score;
                else
                    g_last_liveness = g_last_liveness * 0.7f + real_score * 0.3f;
            }
        }
        g_liveness_ms = (long)(ncnn::get_current_time() - t0);
    }
}

// 消费者线程主循环：从队列取任务处理
static void consumer_loop()
{
    while (true)
    {
        FaceTask task;
        {
            std::unique_lock<std::mutex> lk(g_queue_mutex);
            g_queue_cv.wait(lk, [] { return !g_task_queue.empty() || !g_consumer_running; });
            if (g_task_queue.empty())
                break;  // 收到停止信号且队列已空
            task = std::move(g_task_queue.front());
            g_task_queue.pop();
        }

        g_sem_space.release();  // 取出任务后立即释放空位许可（保证异常/超时路径不泄漏许可）

        g_queue_wait_ms = (long)(ncnn::get_current_time() - task.enqueue_time);  // 队列等待耗时

        process_task(task);
        g_processed_frames++;
    }
}

static void draw_pipeline_result(cv::Mat& rgb)
{
    int y = 20;
    char text[256];

    std::string name;
    float similarity = 0.f;
    float liveness = 0.f;
    bool is_real = false;
    int queue_size = 0;
    {
        ncnn::MutexLockGuard g(lock);
        name = g_last_name;
        similarity = g_last_similarity;
        liveness = g_last_liveness;
        is_real = liveness >= g_liveness_threshold;
    }
    {
        std::lock_guard<std::mutex> lk(g_queue_mutex);
        queue_size = (int)g_task_queue.size();
    }

    // 识别结果（画面用英文，中文名字由 Java TextView 显示，cv::putText 不支持中文）
    if (!name.empty())
        sprintf(text, "recog: known (sim=%.2f)", similarity);
    else
        sprintf(text, "recog: unknown");
    draw_label(rgb, text, y);

    // 活体结果
    sprintf(text, "liveness: %s (%.2f)", is_real ? "real" : "spoof", liveness);
    draw_label(rgb, text, y + 22);

    // 生产者-消费者状态（Semaphore 有界队列）
    sprintf(text, "queue: %d/%d dropped: %d done: %d", queue_size, g_max_queue, g_dropped_frames.load(), g_processed_frames.load());
    draw_label(rgb, text, y + 44);

    // I/O 事件链各阶段耗时
    sprintf(text, "det: %ldms wait: %ldms recog: %ldms live: %ldms", g_detect_ms.load(), g_queue_wait_ms.load(), g_recog_ms.load(), g_liveness_ms.load());
    draw_label(rgb, text, y + 66);
}

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    std::vector<FaceObject> faceobjects;

    // scrfd 检测 + 绘制
    {
        double t0 = ncnn::get_current_time();
        ncnn::MutexLockGuard g(lock);

        if (g_scrfd)
        {
            g_scrfd->detect(rgb, faceobjects);
            g_scrfd->draw(rgb, faceobjects);
        }
        else
        {
            draw_unsupported(rgb);
        }

        g_detect_ms = (long)(ncnn::get_current_time() - t0);
    }

    // 生产者：摄像头采集 -> 选最大脸 -> 放入有界队列（Semaphore 控制，队列满则丢帧）
    if (!faceobjects.empty())
    {
        int best = 0;
        float best_area = -1.f;
        for (size_t i = 0; i < faceobjects.size(); i++)
        {
            float area = faceobjects[i].rect.area();
            if (area > best_area)
            {
                best_area = area;
                best = (int)i;
            }
        }

        if (g_sem_space.try_acquire())
        {
            FaceTask task;
            task.rgb = rgb.clone();
            task.rect = faceobjects[best].rect;
            task.enqueue_time = ncnn::get_current_time();
            memcpy(task.landmark, faceobjects[best].landmark, sizeof(task.landmark));
            {
                std::lock_guard<std::mutex> lk(g_queue_mutex);
                g_task_queue.push(std::move(task));
            }
            g_queue_cv.notify_one();
        }
        else
        {
            g_dropped_frames++;
        }
    }

    draw_pipeline_result(rgb);
    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    // 启动消费者线程（识别/活体生产者-消费者模型）
    g_consumer_running = true;
    g_consumer_thread = std::thread(consumer_loop);

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    // 先停止消费者线程（避免其访问即将释放的模型）
    g_consumer_running = false;
    g_queue_cv.notify_all();
    if (g_consumer_thread.joinable())
        g_consumer_thread.join();

    {
        ncnn::MutexLockGuard g(lock);

        delete g_scrfd;
        g_scrfd = 0;

        delete g_edgeface;
        g_edgeface = 0;

        delete g_minifasnet;
        g_minifasnet = 0;

        g_facestore.clear();
        g_last_feature.clear();
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
    {
        "500m",
        "500m_kps",
        "1g",
        "2.5g",
        "2.5g_kps",
        "10g",
        "10g_kps",
        "34g"
    };

    const char* modeltype = modeltypes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_scrfd;
            g_scrfd = 0;
        }
        else
        {
            if (!g_scrfd)
                g_scrfd = new SCRFD;
            g_scrfd->load(mgr, modeltype, use_gpu);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native boolean loadRecogModel(AssetManager mgr, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadRecogModel(JNIEnv* env, jobject thiz, jobject assetManager, jint cpugpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    {
        ncnn::MutexLockGuard g(lock);

        if (!g_edgeface)
            g_edgeface = new EdgeFace;

        bool use_gpu = (int)cpugpu == 1;
        int ret = g_edgeface->load(mgr, "edgeface_xxs", use_gpu);
        if (ret != 0)
        {
            delete g_edgeface;
            g_edgeface = 0;
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// public native boolean loadLivenessModel(AssetManager mgr, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadLivenessModel(JNIEnv* env, jobject thiz, jobject assetManager, jint cpugpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    {
        ncnn::MutexLockGuard g(lock);

        if (!g_minifasnet)
            g_minifasnet = new MiniFASNet;

        bool use_gpu = (int)cpugpu == 1;
        int ret = g_minifasnet->load(mgr, "minifasnetv2", use_gpu);
        if (ret != 0)
        {
            delete g_minifasnet;
            g_minifasnet = 0;
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// public native boolean registerFace(String name);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_registerFace(JNIEnv* env, jobject thiz, jstring name)
{
    const char* cname = env->GetStringUTFChars(name, 0);
    std::string sname(cname);
    env->ReleaseStringUTFChars(name, cname);

    ncnn::MutexLockGuard g(lock);

    if (g_last_feature.empty())
        return JNI_FALSE;

    return g_facestore.register_face(sname, g_last_feature) ? JNI_TRUE : JNI_FALSE;
}

// public native boolean deleteFace(String name);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_deleteFace(JNIEnv* env, jobject thiz, jstring name)
{
    const char* cname = env->GetStringUTFChars(name, 0);
    std::string sname(cname);
    env->ReleaseStringUTFChars(name, cname);

    ncnn::MutexLockGuard g(lock);

    return g_facestore.remove(sname) ? JNI_TRUE : JNI_FALSE;
}

// public native void clearFaces();
JNIEXPORT void JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_clearFaces(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(lock);
    g_facestore.clear();
}

// public native String getRecogResult();
JNIEXPORT jstring JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getRecogResult(JNIEnv* env, jobject thiz)
{
    char buf[512];
    {
        ncnn::MutexLockGuard g(lock);
        bool is_real = g_last_liveness >= g_liveness_threshold;
        snprintf(buf, sizeof(buf), "识别: %s | 活体: %s (%.2f)",
                 g_last_name.empty() ? "未知" : g_last_name.c_str(),
                 is_real ? "真人" : "攻击", g_last_liveness);
    }
    return env->NewStringUTF(buf);
}

}
