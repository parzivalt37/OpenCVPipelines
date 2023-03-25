import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.CvType;

import java.util.List;
import java.util.ArrayList;

import org.firstinspires.ftc.robotcore.external.Telemetry;

public class ContourDetection extends OpenCvPipeline {
    
    public static int lower;
    public static int higher;

    private Telemetry t;

    public ContourDetection(Telemetry telemetry) {
        t = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2GRAY);
        Mat binary = new Mat(input.rows(), input.cols(), input.type(), new Scalar(0));
        Imgproc.threshold(input, binary, lower, higher, Imgproc.THRESH_BINARY_INV);
        //Finding Contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        //Drawing the Contours
        Scalar color = new Scalar(0, 0, 255);
        Imgproc.drawContours(input, contours, -1, color, 2, Imgproc.LINE_8,
        hierarchy, 2, new Point());
        
        return input;
    }
}
