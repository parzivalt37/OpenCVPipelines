package org.firstinspires.ftc.teamcode;

import org.openftc.easyopencv.OpenCvPipeline;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.Core;

import java.util.ArrayList;

public class CenterstageContourDetection extends OpenCvPipeline {

    private Telemetry t;

    public CenterstageContourDetection(Telemetry telemetry) {
        t = telemetry;
    }


    public Mat processFrame(Mat input) {
        return input;
    }

    


}