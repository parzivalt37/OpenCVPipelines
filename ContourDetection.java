import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.core.RotatedRect;
import org.opencv.core.Rect;
import org.opencv.core.MatOfPoint2f;

import java.util.List;
import java.util.ArrayList;

import org.firstinspires.ftc.robotcore.external.Telemetry;

public class ContourDetection extends OpenCvPipeline {

    //config variables
    public static int binaryLower;
    public static int binaryHigher = 255;
    public static int channelSwitch;

    //Scalars
    public static Scalar lower = new Scalar(17, 72, 144.5);
    public static Scalar upper = new Scalar(51, 255, 255);
    private final Scalar color = new Scalar(255, 0, 255);

    private static List<Mat> channels = new ArrayList<>();
    
    //Mats
    private Mat kernel;
    private Mat grayMat = new Mat();
    private Mat hsvMat = new Mat();
    private Mat thresholdMat_AllContours = new Mat();
    private Mat thresholdMat_MaxContours = new Mat();
    private Mat thresholdMat_BoundingRects = new Mat();
    private Mat thresholdGrayMat = new Mat();
    private Mat maskMat = new Mat();


    private double area;
    private double maxArea = 0;
    private int maxIndex = -1;
    private int processingState = 1;

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
         * 4. Convert the input image to binary.
         * 5. Create an ArrayList to store each of the contours. Contours are each stored in a MatOfPoint, and all contours in an image are stored in that ArrayList.
         * 6. Find the contours from the binary Mats, and draw them on the grayMat and the hsvMat (for consistency across the different Mats). These contours are of the entire
         *    image, not only the thresholded images.
         * 7. Convert the thresholded HSV Mat to grayscale by extracting the thresholded Mat into three Mats and accessing the V mat. This works because the V channel of the HSV
         *    colorspace is already grayscale, so only showing the values of V is the equivalent of converting the whole Mat from RGB to grayscale.
         * 8. Convert the thresholded image to binary.
         * 9. Create an ArrayList to store the contours.
         * 10. Find and draw all of the contours on the thresholdMat_AllContours, and only the largest area contour on thresholdMat_MaxContours
         * 11. Change which Mat is returned to the viewport based on EOCV-Sim variable tuners
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
        Core.split(thresholdMat_AllContours, channels);
        thresholdGrayMat = channels.get(2);

        //8. Convert to binary
        contourBinary = new Mat(thresholdMat_AllContours.rows(), thresholdMat_AllContours.cols(), thresholdMat_AllContours.type(), new Scalar(0));
        Imgproc.threshold(thresholdGrayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        //9. ArrayList storing contours
        List<MatOfPoint> thresholdContours = new ArrayList<>();
        Mat thresholdHierarchy = new Mat();

        //10. Finding/drawing contours
        Imgproc.findContours(contourBinary, thresholdContours, thresholdHierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        
        
        thresholdMat_MaxContours = thresholdMat_AllContours.clone();
        thresholdMat_BoundingRects = thresholdMat_AllContours.clone();
        
        for(int i = 0; i < thresholdContours.size(); i++) {
            MatOfPoint contour = thresholdContours.get(i);
            area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
            MatOfPoint2f temp = new MatOfPoint2f(contour.toArray());
            
            RotatedRect r =  Imgproc.minAreaRect(temp);
            Point vertices[] = new Point[4];
            r.points(vertices);

            for(int j = 0; j < 4; j++) {
                Imgproc.line(thresholdMat_BoundingRects, vertices[j], vertices[(j+1) % 4], color, 2);
            }
            temp.release();
        }

        Imgproc.drawContours(thresholdMat_AllContours, thresholdContours, -1, color, 2);
        Imgproc.drawContours(thresholdMat_MaxContours, thresholdContours, maxIndex, color, 2);

        t.addData("Maximum contour area: ", maxArea);
        t.update();

        //11. Return processed Mats
        switch(channelSwitch) {
            case 1:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                thresholdMat_AllContours.release();
                thresholdMat_MaxContours.release();
                thresholdMat_BoundingRects.release();
                thresholdGrayMat.release();
                maskMat.release();
                
                return input;
            case 2:
                kernel.release();
                input.release();
                hsvMat.release();
                thresholdMat_AllContours.release();
                thresholdMat_MaxContours.release();
                thresholdMat_BoundingRects.release();
                thresholdGrayMat.release();
                maskMat.release();    
                
                return grayMat;
            case 3:
                kernel.release();
                grayMat.release();
                input.release();
                thresholdMat_AllContours.release();
                thresholdMat_MaxContours.release();
                thresholdMat_BoundingRects.release();
                thresholdGrayMat.release();
                maskMat.release();
                
                return hsvMat;
            case 5:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                input.release();
                thresholdMat_AllContours.release();
                thresholdMat_BoundingRects.release();
                thresholdGrayMat.release();
                maskMat.release();

                return thresholdMat_MaxContours;
            case 6:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                input.release();
                thresholdMat_AllContours.release();
                thresholdMat_MaxContours.release();
                thresholdGrayMat.release();
                maskMat.release();

                return thresholdMat_BoundingRects;
            case 4:
            default:
                kernel.release();
                grayMat.release();
                hsvMat.release();
                input.release();
                thresholdGrayMat.release();
                thresholdMat_MaxContours.release();
                thresholdMat_BoundingRects.release();
                maskMat.release();
                
                return thresholdMat_AllContours;
        }
    }

    private void thresholdHSV(Mat input) {
        Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_RGB2HSV);
        Core.inRange(hsvMat, lower, upper, maskMat);
        thresholdMat_AllContours.release();
        Core.bitwise_and(hsvMat, hsvMat, thresholdMat_AllContours, maskMat);
    }
}