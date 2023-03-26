import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.Core;

import java.util.List;
import java.util.ArrayList;

import org.firstinspires.ftc.robotcore.external.Telemetry;

public class ContourDetection extends OpenCvPipeline {
    
    //config variables
    public static int binaryLower;
    public static int binaryHigher = 255;
    public static int channelSwitch;

    //Scalars
    public static Scalar lower = new Scalar(0, 0, 0);
    public static Scalar upper = new Scalar(255, 255, 255);
    private final Scalar color = new Scalar(0, 0, 255);    
    
    //Mats
    private Mat kernel;
    private Mat grayMat = new Mat();
    private Mat hsvMat = new Mat();
    private Mat thresholdMat = new Mat();
    private Mat maskMat = new Mat();

    private Telemetry t;

    public ContourDetection(Telemetry telemetry) {
        t = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        //Morphology to tune out noise
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(input, input, Imgproc.MORPH_CLOSE, kernel);

        thresholdHSV(input);
        
        Imgproc.cvtColor(input, grayMat, Imgproc.COLOR_RGB2GRAY);
        
        Mat contourBinary = new Mat(input.rows(), input.cols(), input.type(), new Scalar(0));
        Imgproc.threshold(grayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        //Finding Contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(contourBinary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.drawContours(grayMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());
        Imgproc.drawContours(hsvMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());


        //Finding contours of the thresholded Mat
        List<MatOfPoint> thresholdContours = new ArrayList<>();
        Mat thresholdHierarchy = new Mat();
        Imgproc.findContours(contourBinary, thresholdContours, thresholdHierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(thresholdMat, thresholdContours, -1, color, 2, Imgproc.LINE_8, thresholdHierarchy, 2, new Point());
        
        
        switch(channelSwitch) {
            case 1:
            default:
                return input;
            case 2:
                return grayMat;
            case 3:
                return hsvMat;
            case 4:
                return thresholdMat;
        }
    }

    private void thresholdHSV(Mat input) {
        Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_RGB2HSV);
        Core.inRange(hsvMat, lower, upper, maskMat);
        thresholdMat.release();
        Core.bitwise_and(hsvMat, hsvMat, thresholdMat, maskMat);
    }
}