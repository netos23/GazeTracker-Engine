#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv/cv_image.h"
#include <dlib/image_processing.h>
#include <iostream>

using namespace cv;
using namespace dlib;

void detectAndDisplay(Mat frame);

frontal_face_detector hogFaceDetector;

int main(int argc, const char **argv) {
    float temp = 0;
    if (temp != 0 && fpclassify(temp) != FP_SUBNORMAL) {}
    else {

    }
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{camera|0|Camera device number.}");
    parser.about(
            "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
            "You can use Haar or LBP features.\n\n");
    parser.printMessage();
    hogFaceDetector = get_frontal_face_detector();

    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened()) {
        std::cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            std::cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);
        if (waitKey(10) == 27) {
            break; // escape
        }
    }

    return 0;
}

void detectAndDisplay(Mat frame) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    cv_image<bgr_pixel> dlibIm(frame);
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

    for (auto &faceRect: faceRects) {

        long x1 = faceRect.left();
        long y1 = faceRect.top();
        long x2 = faceRect.right();
        long y2 = faceRect.bottom();

        cv::rectangle(frame, Point(x1, y1), Point(x2, y2),
                      Scalar(0, 255, 0), 1, 4);

    }


    //-- Show what you got
    imshow("Capture - Face detection", frame);
}