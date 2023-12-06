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

package org.firstinspires.ftc.teamcode;

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
         * 
         */
        
        // 1. Apply morphology
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(input, input, Imgproc.MORPH_CLOSE, kernel);

        // 2. Apply the threshold to the HSV mat
        thresholdHSV(input);

        // 3. Convert the input Mat to grayscale
        Imgproc.cvtColor(input, grayMat, Imgproc.COLOR_RGB2GRAY);
        
        // 4. Conver the input Mat to binary
        Mat contourBinary = new Mat(input.rows(), input.cols(), input.type(), new Scalar(0));
        Imgproc.threshold(grayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        // 5. Create an ArrayList storing all of the contours in the binary input Mat
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        
        // 6. Finding/drawing contours on both the grayscale and HSV Mats
        Imgproc.findContours(contourBinary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(grayMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());
        Imgproc.drawContours(hsvMat, contours, -1, color, 2, Imgproc.LINE_8, hierarchy, 2, new Point());



        // 7. Extract the V channel from the thresholded HSV Mat
        Core.split(thresholdMat_AllContours, channels); //At this point, thresholdMat_AllContours has no contours drawn on it, it's just the thresholded HSV Mat
        thresholdGrayMat = channels.get(2); //Thresholded grayscale Mat

        //8. Convert the thresholded HSV mat to binary
        contourBinary = new Mat(thresholdMat_AllContours.rows(), thresholdMat_AllContours.cols(), thresholdMat_AllContours.type(), new Scalar(0));
        Imgproc.threshold(thresholdGrayMat, contourBinary, binaryLower, binaryHigher, Imgproc.THRESH_BINARY_INV);
        
        //9. Create another ArrayList storing the contours detected in the thresholded Mat
        List<MatOfPoint> thresholdContours = new ArrayList<>();
        Mat thresholdHierarchy = new Mat();

        //10. Find and store contours detected in thresholded Mat
        Imgproc.findContours(contourBinary, thresholdContours, thresholdHierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        
        //11. Make the MaxContours and BoundingRects Mats match the AllContours Mat
        thresholdMat_MaxContours = thresholdMat_AllContours.clone();
        thresholdMat_BoundingRects = thresholdMat_AllContours.clone();
        
        //12. For loop identifying the contour of the greatest area
        for(int i = 0; i < thresholdContours.size(); i++) { //Can't be a foreach loop since foreach elements are immutable
            MatOfPoint contour = thresholdContours.get(i);
            area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
        }

        //13. Resets the value of the maxArea and removes the contour of largest area. By default, the largest contour will be the border of the image, so by removing it from the
        //contours ArrayList this allows us to focus on the other contours inside the image.
        maxArea = 0;
        thresholdContours.remove(maxIndex);

        //14. Identifies the new largest contour inside the Mat and draws BoundedRects (as a series of 4 lines) on the BoundedRects Mat
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

        //15. Draws contours on the AllContours and MaxContours Mat
        Imgproc.drawContours(thresholdMat_AllContours, thresholdContours, -1, color, 2);
        Imgproc.drawContours(thresholdMat_MaxContours, thresholdContours, maxIndex, color, 2);

        t.addData("Maximum contour area: ", maxArea);
        t.update();

        //16. Return processed Mats. All unreturned Mats are released to keep memory clean.
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