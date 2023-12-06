import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.openftc.easyopencv.OpenCvPipeline;

package org.firstinspires.ftc.teamcode;

public class CannyDetectionPipeline extends OpenCvPipeline {
    
    public static int lower;
    public static int higher;

    private Mat cannyMat = new Mat();
    
    private Telemetry t;

    public CannyDetectionPipeline(Telemetry telemetry) {
        t = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.Canny(input, cannyMat, lower, higher);
        return cannyMat;
    }
}