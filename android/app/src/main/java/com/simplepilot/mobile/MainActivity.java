package com.simplepilot.mobile;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.view.WindowManager;

import com.simplepilot.mobile.sensor.CameraManager;

public class MainActivity extends AppCompatActivity {

    public static Context appContext;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        appContext = getApplicationContext();

        // keep app from dimming due to inactivity.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        CameraManager cameraManager = new CameraManager(appContext, 20, "roadCameraState");
        cameraManager.start();
        String modelPath = "/storage/emulated/0/Android/data/ai.flow.android/files/supercombo";


    }
}