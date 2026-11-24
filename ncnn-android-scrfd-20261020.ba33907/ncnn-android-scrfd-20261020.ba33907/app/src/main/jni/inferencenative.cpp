// 课程综合实验：独立推理 JNI（供双进程 AI 推理服务 InferenceService 使用）
// 复用 edgeface / minifasnet / facestore，不依赖摄像头。

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include <android/log.h>

#include <string>
#include <vector>

#include <platform.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "edgeface.h"
#include "minifasnet.h"
#include "facestore.h"

// 注意：这里的全局变量是 static（文件作用域），与 scrfdncnn.cpp 里的同名变量互不影响
static EdgeFace* g_edgeface = 0;
static MiniFASNet* g_minifasnet = 0;
static FaceStore g_facestore;
static ncnn::Mutex g_lock;

extern "C" {

// public static native boolean loadModels(AssetManager mgr, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_NativeInference_loadModels(JNIEnv* env, jclass thiz, jobject assetManager, jint cpugpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    ncnn::MutexLockGuard g(g_lock);

    if (!g_edgeface)
        g_edgeface = new EdgeFace;
    if (!g_minifasnet)
        g_minifasnet = new MiniFASNet;

    bool use_gpu = (int)cpugpu == 1;
    int r1 = g_edgeface->load(mgr, "edgeface_xxs", use_gpu);
    int r2 = g_minifasnet->load(mgr, "minifasnetv2", use_gpu);

    return (r1 == 0 && r2 == 0) ? JNI_TRUE : JNI_FALSE;
}

// public static native float[] extractFeature(byte[] rgb, int width, int height, float[] landmark);
JNIEXPORT jfloatArray JNICALL Java_com_tencent_scrfdncnn_NativeInference_extractFeature(JNIEnv* env, jclass thiz, jbyteArray rgb, jint width, jint height, jfloatArray landmark)
{
    // rgb byte[] -> cv::Mat
    jbyte* rgbData = env->GetByteArrayElements(rgb, 0);
    cv::Mat img(height, width, CV_8UC3, (unsigned char*)rgbData);
    cv::Mat imgCopy = img.clone();
    env->ReleaseByteArrayElements(rgb, rgbData, JNI_ABORT);

    // landmark -> cv::Point2f[5]
    jfloat* lm = env->GetFloatArrayElements(landmark, 0);
    cv::Point2f pts[5];
    for (int i = 0; i < 5; i++)
        pts[i] = cv::Point2f(lm[i * 2], lm[i * 2 + 1]);
    env->ReleaseFloatArrayElements(landmark, lm, JNI_ABORT);

    std::vector<float> feature;
    {
        ncnn::MutexLockGuard g(g_lock);

        if (!g_edgeface || !g_edgeface->is_loaded())
            return 0;

        cv::Mat aligned;
        if (g_edgeface->align_face(imgCopy, pts, aligned) != 0)
            return 0;
        if (g_edgeface->extract_feature(aligned, feature) != 0)
            return 0;
    }

    jfloatArray result = env->NewFloatArray((jsize)feature.size());
    env->SetFloatArrayRegion(result, 0, (jsize)feature.size(), feature.data());
    return result;
}

// public static native float detectLiveness(byte[] rgb, int width, int height);
JNIEXPORT jfloat JNICALL Java_com_tencent_scrfdncnn_NativeInference_detectLiveness(JNIEnv* env, jclass thiz, jbyteArray rgb, jint width, jint height)
{
    jbyte* rgbData = env->GetByteArrayElements(rgb, 0);
    cv::Mat img(height, width, CV_8UC3, (unsigned char*)rgbData);
    cv::Mat imgCopy = img.clone();
    env->ReleaseByteArrayElements(rgb, rgbData, JNI_ABORT);

    float score = 0.f;
    {
        ncnn::MutexLockGuard g(g_lock);

        if (!g_minifasnet || !g_minifasnet->is_loaded())
            return 0.f;

        g_minifasnet->detect_liveness(imgCopy, score);
    }
    return score;
}

// public static native boolean registerFace(String name, float[] feature);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_NativeInference_registerFace(JNIEnv* env, jclass thiz, jstring name, jfloatArray feature)
{
    const char* cname = env->GetStringUTFChars(name, 0);
    std::string sname(cname);
    env->ReleaseStringUTFChars(name, cname);

    jfloat* feat = env->GetFloatArrayElements(feature, 0);
    jsize len = env->GetArrayLength(feature);
    std::vector<float> f(feat, feat + len);
    env->ReleaseFloatArrayElements(feature, feat, JNI_ABORT);

    ncnn::MutexLockGuard g(g_lock);
    return g_facestore.register_face(sname, f) ? JNI_TRUE : JNI_FALSE;
}

// public static native String recognizeFace(float[] feature, float threshold);
JNIEXPORT jstring JNICALL Java_com_tencent_scrfdncnn_NativeInference_recognizeFace(JNIEnv* env, jclass thiz, jfloatArray feature, jfloat threshold)
{
    jfloat* feat = env->GetFloatArrayElements(feature, 0);
    jsize len = env->GetArrayLength(feature);
    std::vector<float> f(feat, feat + len);
    env->ReleaseFloatArrayElements(feature, feat, JNI_ABORT);

    std::string name;
    {
        ncnn::MutexLockGuard g(g_lock);
        float score = 0.f;
        name = g_facestore.recognize(f, (float)threshold, score);
    }
    return env->NewStringUTF(name.c_str());
}

// public static native boolean deleteFace(String name);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_NativeInference_deleteFace(JNIEnv* env, jclass thiz, jstring name)
{
    const char* cname = env->GetStringUTFChars(name, 0);
    std::string sname(cname);
    env->ReleaseStringUTFChars(name, cname);

    ncnn::MutexLockGuard g(g_lock);
    return g_facestore.remove(sname) ? JNI_TRUE : JNI_FALSE;
}

// public static native void clearFaces();
JNIEXPORT void JNICALL Java_com_tencent_scrfdncnn_NativeInference_clearFaces(JNIEnv* env, jclass thiz)
{
    ncnn::MutexLockGuard g(g_lock);
    g_facestore.clear();
}

} // extern "C"
