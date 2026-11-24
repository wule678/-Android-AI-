package com.tencent.scrfdncnn;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.Process;
import android.util.Log;

// AI 推理服务：运行在独立进程 :inference，通过 Binder(AIDL) 对外提供推理接口。
// 识别/活体/人脸库都跑在这个进程，与主进程（UI + 摄像头）隔离。
public class InferenceService extends Service
{
    private static final String TAG = "InferenceService";

    // IPC 请求统计
    private long requestCount = 0;
    private long totalLatencyMs = 0;

    private final IInferenceService.Stub binder = new IInferenceService.Stub()
    {
        @Override
        public boolean loadModels()
        {
            return NativeInference.loadModels(getAssets(), 0);
        }

        @Override
        public int getPid()
        {
            return Process.myPid();
        }

        @Override
        public float[] extractFeature(byte[] rgb, int width, int height, float[] landmark)
        {
            long t0 = System.currentTimeMillis();
            try {
                return NativeInference.extractFeature(rgb, width, height, landmark);
            } finally {
                record(t0);
            }
        }

        @Override
        public float detectLiveness(byte[] rgb, int width, int height)
        {
            long t0 = System.currentTimeMillis();
            try {
                return NativeInference.detectLiveness(rgb, width, height);
            } finally {
                record(t0);
            }
        }

        @Override
        public boolean registerFace(String name, float[] feature)
        {
            return NativeInference.registerFace(name, feature);
        }

        @Override
        public String recognizeFace(float[] feature, float threshold)
        {
            return NativeInference.recognizeFace(feature, threshold);
        }

        @Override
        public boolean deleteFace(String name)
        {
            return NativeInference.deleteFace(name);
        }

        @Override
        public void clearFaces()
        {
            NativeInference.clearFaces();
        }

        @Override
        public long getRequestCount()
        {
            return requestCount;
        }

        @Override
        public long getTotalLatencyMs()
        {
            return totalLatencyMs;
        }

        private void record(long t0)
        {
            requestCount++;
            totalLatencyMs += (System.currentTimeMillis() - t0);
        }
    };

    @Override
    public void onCreate()
    {
        super.onCreate();
        Log.i(TAG, "onCreate pid=" + Process.myPid());
        // 启动即加载模型（失败不影响服务存活，接口会返回错误）
        boolean ok = NativeInference.loadModels(getAssets(), 0);
        Log.i(TAG, "loadModels=" + ok);
    }

    @Override
    public IBinder onBind(Intent intent)
    {
        Log.i(TAG, "onBind pid=" + Process.myPid());
        return binder;
    }
}
