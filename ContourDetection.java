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
    private final Scalar color = new Scalar(255, 0, 255);

    private static List<Mat> channels = new ArrayList<>();
    
    //Mats
    private Mat kernel;
    private Mat grayMat = new Mat();
    private Mat hsvMat = new Mat();
    private Mat thresholdMat = new Mat();
    private Mat thresholdGrayMat = new Mat();
    private Mat maskMat = new Mat();
    private double maxArea = 0;
    private MatOfPoint maxAreaMOP;

    private Telemetry t;

    public ContourDetection(Telemetry telemetry) {
        t = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        /*
         * The general structure of this pipeline:
         * 1. Apply morphology to the input image. This tunes out some of the "noise" of the image that makes it easier to detect contours and edges.
         * 2. Apply the threshold from EOCV-Sim variable tuners to the HSV image.
         * 3. Convert the input image to grayscale and store it in grayMat. This is so that the contour detector can accurately detect contours
         * 4. Convert the input image to binary. I'm not entirely sure what this does, but I just know that it's necessary for contour detection.
         * 5. Create an ArrayList to store each of the contours. Contours are each stored in a MatOfPoint, and all contours in an image are stored in that ArrayList.
         * 6. Find the contours from the binary Mats, and draw them on the grayMat and the hsvMat (for consistency across the different Mats). These contours are of the entire
         *    image, not only the thresholded images.
         * 7. Convert the thresholded HSV Mat to grayscale by extracting the thresholded Mat into three Mats and accessing the V mat. This works because the V channel of the HSV
         *    colorspace is already grayscale, so only showing the values of V is the equivalent of converting the whole Mat from RGB to grayscale.
         * 8. Convert the thresholded image to binary.
         * 9. Create an ArrayList to store the contours.
         * 10. Find and draw contours and draw them on the thresholdMat.
         * 11. Find the biggest area of the thresholdMat contours from the thresholdContours ArrayList.
         * 12. Change which Mat is returned to the viewport based on EOCV-Sim variable tuners
         */
        
        // 1. Morphology
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(input, input, Imgproc.MORPH_CLOSE, kernel);

        // 2. Threshold HSV
        thresholdHSV(input);

        // 3. Convert to grayscale
        Imgproc.cvtColor(input, grayMat, Imgproc.COLOR_RGB2GRAY);
        
        // 4. Convert to binary
        Mat contourBinary = new Mat(input.rows(), input.cols(), input.type(), new Scalar(0));
        Imgproc.threshold(grayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        // 5. ArrayList storing contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        
        // 6. Finding/drawing contours
        Imgproc.findContours(contourBinary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(grayMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());
        Imgproc.drawContours(hsvMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());


        // 7. Extract the V channel
        Core.split(thresholdMat, channels);
        thresholdGrayMat = channels.get(2);

        //8. Convert to binary
        contourBinary = new Mat(thresholdMat.rows(), thresholdMat.cols(), thresholdMat.type(), new Scalar(0));
        Imgproc.threshold(thresholdGrayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        //9. ArrayList storing contours
        List<MatOfPoint> thresholdContours = new ArrayList<>();
        Mat thresholdHierarchy = new Mat();

        //10. Finding/drawing contours
        Imgproc.findContours(contourBinary, thresholdContours, thresholdHierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(thresholdMat, thresholdContours, -1, color, 2, Imgproc.LINE_8, thresholdHierarchy, 2, new Point());
        
        //11. Find largest contour on the thresholdMat
        for(MatOfPoint contour : thresholdContours) {
            if (Imgproc.contourArea(contour) > maxArea) {
                maxArea = Imgproc.contourArea(contour);
                maxAreaMOP = contour;
            }
        }

        t.addData("Maximum contour area: ", maxArea);
        t.update();

        //12. Return processed Mats
        switch(channelSwitch) {
            case 1:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                thresholdMat.release();
                thresholdGrayMat.release();
                maskMat.release();
                
                return input;
            case 2:
                kernel.release();
                input.release();
                hsvMat.release();
                thresholdMat.release();
                thresholdGrayMat.release();
                maskMat.release();    
                
                return grayMat;
            case 3:
                kernel.release();
                grayMat.release();
                input.release();
                thresholdMat.release();
                thresholdGrayMat.release();
                maskMat.release();
                
                return hsvMat;
            case 4:
            default:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                input.release();
                thresholdGrayMat.release();
                maskMat.release();
                
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