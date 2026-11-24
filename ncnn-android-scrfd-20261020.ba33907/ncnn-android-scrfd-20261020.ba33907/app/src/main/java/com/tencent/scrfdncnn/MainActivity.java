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

package com.tencent.scrfdncnn;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private SCRFDNcnn scrfdncnn = new SCRFDNcnn();
    private int facing = 0;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;

    private IInferenceService inferenceService = null;
    private boolean inferenceBound = false;
    private TextView textInference;
    private TextView textResult;
    private Handler handler = new Handler();
    private Runnable updateResultRunnable = new Runnable() {
        @Override
        public void run() {
            if (textResult != null) {
                textResult.setText(scrfdncnn.getRecogResult());
            }
            handler.postDelayed(this, 300);
        }
    };

    private ServiceConnection inferenceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            inferenceService = IInferenceService.Stub.asInterface(service);
            inferenceBound = true;
            int servicePid = 0;
            try {
                servicePid = inferenceService.getPid();
            } catch (Exception e) {
            }
            Log.i("MainActivity", "推理服务已连接, 本进程pid=" + android.os.Process.myPid() + ", 推理进程pid=" + servicePid);
            if (textInference != null) {
                textInference.setText("本进程 PID: " + android.os.Process.myPid() + "  |  推理进程 PID: " + servicePid);
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            inferenceService = null;
            inferenceBound = false;
            Log.w("MainActivity", "推理服务断开");
            if (textInference != null) {
                textInference.setText("推理服务断开");
            }
        }
    };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                scrfdncnn.closeCamera();

                scrfdncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setSelection(1);   // 默认选 500m_kps（带 5 关键点，识别对齐必需）
        current_model = 1;
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        Button buttonRegister = (Button) findViewById(R.id.buttonRegister);
        buttonRegister.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final EditText input = new EditText(MainActivity.this);
                input.setHint("输入姓名");
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("注册人脸")
                        .setMessage("请把脸对准摄像头，输入姓名后点确定")
                        .setView(input)
                        .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                String name = input.getText().toString().trim();
                                if (name.isEmpty()) {
                                    Toast.makeText(MainActivity.this, "姓名不能为空", Toast.LENGTH_SHORT).show();
                                    return;
                                }
                                boolean ok = scrfdncnn.registerFace(name);
                                Toast.makeText(MainActivity.this, ok ? ("已注册: " + name) : "注册失败（请对准人脸，且姓名不能重复）", Toast.LENGTH_SHORT).show();
                            }
                        })
                        .setNegativeButton("取消", null)
                        .show();
            }
        });

        Button buttonDelete = (Button) findViewById(R.id.buttonDelete);
        buttonDelete.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final EditText input = new EditText(MainActivity.this);
                input.setHint("输入姓名");
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("删除人脸")
                        .setView(input)
                        .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                String name = input.getText().toString().trim();
                                if (name.isEmpty()) {
                                    Toast.makeText(MainActivity.this, "姓名不能为空", Toast.LENGTH_SHORT).show();
                                    return;
                                }
                                boolean ok = scrfdncnn.deleteFace(name);
                                Toast.makeText(MainActivity.this, ok ? ("已删除: " + name) : "删除失败（未找到该姓名）", Toast.LENGTH_SHORT).show();
                            }
                        })
                        .setNegativeButton("取消", null)
                        .show();
            }
        });

        Button buttonClear = (Button) findViewById(R.id.buttonClear);
        buttonClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                scrfdncnn.clearFaces();
                Toast.makeText(MainActivity.this, "已清空人脸库", Toast.LENGTH_SHORT).show();
            }
        });

        textInference = (TextView) findViewById(R.id.textInference);
        textResult = (TextView) findViewById(R.id.textResult);

        // 绑定独立进程的 AI 推理服务（Binder IPC）
        Intent intent = new Intent(MainActivity.this, InferenceService.class);
        bindService(intent, inferenceConnection, Context.BIND_AUTO_CREATE);

        reload();
    }

    private void reload()
    {
        boolean ret_init = scrfdncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "scrfdncnn loadModel failed");
        }

        // 识别模型（EdgeFace-XXS）+ 活体模型（MiniFASNetV2）
        // 模型文件缺失时会返回 false，App 仍可只做检测，不崩溃
        boolean ret_recog = scrfdncnn.loadRecogModel(getAssets(), current_cpugpu);
        if (!ret_recog)
        {
            Log.e("MainActivity", "loadRecogModel failed（请确认 assets 里有 edgeface_xxs.param/.bin）");
        }

        boolean ret_liveness = scrfdncnn.loadLivenessModel(getAssets(), current_cpugpu);
        if (!ret_liveness)
        {
            Log.e("MainActivity", "loadLivenessModel failed（请确认 assets 里有 minifasnetv2.param/.bin）");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        scrfdncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        scrfdncnn.openCamera(facing);

        handler.post(updateResultRunnable);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        scrfdncnn.closeCamera();

        handler.removeCallbacks(updateResultRunnable);
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();

        if (inferenceBound && inferenceConnection != null) {
            unbindService(inferenceConnection);
            inferenceBound = false;
        }
    }
}
