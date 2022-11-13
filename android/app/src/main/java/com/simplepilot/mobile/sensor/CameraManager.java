package com.simplepilot.mobile.sensor;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Rect;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.util.Log;
import android.util.Range;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.camera.camera2.interop.Camera2Interop;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.LifecycleRegistry;

import com.google.common.util.concurrent.ListenableFuture;
import com.simplepilot.mobile.selfdrive.sensor.SensorInterface;

import org.capnproto.PrimitiveList;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.concurrent.ExecutionException;

public class CameraManager extends SensorInterface {

    public ProcessCameraProvider cameraProvider;
    public Mat frame, frameCrop, frameCropContinuous;
    public String topic;
    public boolean running = false;
    public int defaultFrameWidth = 1164;
    public int defaultFrameHeight = 874;
    public org.opencv.core.Size sz = new org.opencv.core.Size(defaultFrameWidth, defaultFrameHeight);
    public int frequency;
    public int frameID = 0;
    public boolean recording = false;
    public Context context;



    public CameraManager(Context context, int frequency, String topic){
        this.context = context;
        this.frequency = frequency;
        this.topic = topic;
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        frame = new Mat();
        frameCropContinuous = new Mat(874, 1164, CvType.CV_8UC3);
    }


    static class CustomLifecycle implements LifecycleOwner {

        private final LifecycleRegistry mLifecycleRegistry;

        CustomLifecycle() {
            mLifecycleRegistry = new LifecycleRegistry(this);
            mLifecycleRegistry.setCurrentState(Lifecycle.State.CREATED);
        }

        void doOnResume() {
            mLifecycleRegistry.setCurrentState(Lifecycle.State.RESUMED);
        }

        void doOnStart() {
            mLifecycleRegistry.setCurrentState(Lifecycle.State.STARTED);
        }

        @NonNull
        public Lifecycle getLifecycle() {
            return mLifecycleRegistry;
        }

    }

    public void start() {
        if (running)
            return;
        running = true;
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {

                try {
                    cameraProvider = cameraProviderFuture.get();
                    bindImageAnalysis(cameraProvider);
                    Log.d("CameraManager", "awd");

                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }
        }, ContextCompat.getMainExecutor(context));
    }

    @SuppressLint({"RestrictedApi", "UnsafeOptInUsageError"})
    private void bindImageAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        builder.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST);
        builder.setTargetResolution(new Size(1280, 960));
        Camera2Interop.Extender<ImageAnalysis> ext = new Camera2Interop.Extender<>(builder);
        ext.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(frequency, frequency));
        ImageAnalysis imageAnalysis = builder.build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context), new ImageAnalysis.Analyzer() {
            @RequiresApi(api = Build.VERSION_CODES.N)
            @Override
            public void analyze(@NonNull ImageProxy image) {
//                ImUtils.Image2RGB(image, frame);
//                if (frameCrop==null) {
//                    frameCrop = frame.submat(new Rect((frame.width() - defaultFrameWidth) / 2, (frame.height() - defaultFrameHeight) / 2,
//                            defaultFrameWidth, defaultFrameHeight));
//                }
//                frameCrop.copyTo(frameCropContinuous); // make sub-mat continuous.
//                msgFrameData.frameData.setFrameId(frameID);
//                ph.publishBuffer(topic, msgFrameData.serialize());
                image.close();
                //frameID += 1;
            }
        });



        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

        CustomLifecycle lifecycle=new CustomLifecycle();
        lifecycle.doOnResume();
        lifecycle.doOnStart();
        cameraProvider.bindToLifecycle(lifecycle, cameraSelector,
                imageAnalysis, null);
    }


    public boolean isRunning() {
        return running;
    }

    @SuppressLint("RestrictedApi")
    @Override
    public void stop() {
//        videoCapture.stopRecording();
//        cameraProvider.unbindAll();
//        ph.releaseAll();
//        frame.release();
//        frameCrop.release();
        running = false;
    }

}
