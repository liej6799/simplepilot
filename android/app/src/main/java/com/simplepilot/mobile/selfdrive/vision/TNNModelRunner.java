package com.simplepilot.mobile.selfdrive.vision;
// https://github.com/flowdriveai/flowpilot/blob/master/selfdrive/vision/java/ai.flow.vision/TNNModelRunner.java

import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class TNNModelRunner extends ModelRunner{
    String modelPath;
    String deviceType;
    TNN model;
    Map<String, ByteBuffer> container = new HashMap<>();
    boolean useGPU;

    public TNNModelRunner(String modelPath, boolean useGPU){
        this.modelPath = modelPath;
        this.useGPU = useGPU;
    }

    @Override
    public void init(Map<String, int[]> shapes){
        System.loadLibrary("tnnjni");
        model = new TNN();

        // ARM Android only, might changed if need to support x86 android
        deviceType = "ARM";


        model.init(modelPath, deviceType);

        model.createInput("input_imgs", shapes.get("input_imgs"));
        model.createInput("initial_state", shapes.get("initial_state"));
        model.createInput("desire", shapes.get("desire"));
        model.createInput("traffic_convention", shapes.get("traffic_convention"));
        model.createOutput("outputs", shapes.get("outputs"));
    }

    @Override
    public void run(ByteBuffer inputImgs, ByteBuffer desire, ByteBuffer trafficConvention, ByteBuffer state, float[] netOutputs) {
        container.put("input_imgs", inputImgs);
        container.put("desire", desire);
        container.put("traffic_convention", trafficConvention);
        container.put("initial_state", state);

        model.forward(container, "outputs", netOutputs);
    }

    @Override
    public void run(INDArray inputImgs, INDArray desire, INDArray trafficConvention, INDArray state, float[] netOutputs) {
        container.put("input_imgs", inputImgs.data().asNio());
        container.put("desire", desire.data().asNio());
        container.put("traffic_convention", trafficConvention.data().asNio());
        container.put("initial_state", state.data().asNio());

        model.forward(container, "outputs", netOutputs);
    }
}

