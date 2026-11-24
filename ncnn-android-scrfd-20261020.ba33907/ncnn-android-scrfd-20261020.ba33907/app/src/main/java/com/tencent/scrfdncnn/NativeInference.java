package com.tencent.scrfdncnn;

import android.content.res.AssetManager;

// 独立推理 JNI 封装：供双进程 AI 推理服务（InferenceService）使用。
// 与摄像头无关，只负责 EdgeFace / MiniFASNet / 人脸库。
public class NativeInference
{
    public static native boolean loadModels(AssetManager mgr, int cpugpu);
    public static native float[] extractFeature(byte[] aligned, int width, int height);
    public static native float detectLiveness(byte[] rgb, int width, int height);
    public static native boolean registerFace(String name, float[] feature);
    public static native String recognizeFace(float[] feature, float threshold);
    public static native boolean deleteFace(String name);
    public static native void clearFaces();

    static {
        System.loadLibrary("scrfdncnn");
    }
}
