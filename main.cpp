#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv/cv_image.h"
#include <dlib/image_processing.h>
#include <iostream>
#include "GazeTracker.h"

#define RED_COLOR Scalar(0, 0, 255)
#define GREEN_COLOR Scalar(0, 255, 0)
#define BLUE_COLOR Scalar(255, 0, 0)
using namespace cv;
using namespace dlib;

void detectAndDisplay(Mat frame);

void drawFaces(const Mat &frame, const std::vector<dlib::rectangle> &faceRects);

void drawLandmarks(const Mat &frame, const std::vector<dlib::full_object_detection> &facesLandmarks);

void getGazeDetector(GazeTracker &gazeTracker);

void drawFps(const Mat &frame);

void drawText(const Mat &frame, const String &text, Point position);

double timer;

int main(int argc, const char **argv) {
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{camera|0|Camera device number.}");
    parser.about(
            "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
            "You can use Haar or LBP features.\n\n");
    parser.printMessage();


    GazeTracker gazeTracker;
    getGazeDetector(gazeTracker);


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
        timer = (double) getTickCount();

        gazeTracker.update(frame);

        std::vector<dlib::rectangle> rects = gazeTracker.getFacesRect();
        drawFaces(frame, rects);
        std::vector<dlib::full_object_detection> landmarks = gazeTracker.getFacesLandmarks();
        drawLandmarks(frame, landmarks);


        if (gazeTracker.isBlinking(LEFT_EYE)) {
            drawText(frame, "LEFT_EYE_BLINKING", Point(100, 100));
        }
        if (gazeTracker.isBlinking(RIGHT_EYE)) {
            drawText(frame, "RIGHT_EYE_BLINKING", Point(100, 150));
        }
        drawFps(frame);
        imshow("Capture - Face detection", frame);

        if (waitKey(10) == 27) {
            break; // escape
        }
    }

    return 0;
}

void drawFps(const Mat &frame) {

    float fps = getTickFrequency() / ((double) getTickCount() - timer);
    String text = "FPS : " + std::to_string(int(fps));
    drawText(frame, text, Point(100, 50));
}

void drawText(const Mat &frame, const String &text, Point position) {
    putText(frame, text, position,
            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
}

void getGazeDetector(GazeTracker &gazeTracker) {
    frontal_face_detector hogFaceDetector = get_frontal_face_detector();
    shape_predictor shapePredictor;
    deserialize("data/shape_predictor_68_face_landmarks.dat") >> shapePredictor;
    Ptr<TrackerMOSSE> tracker = TrackerMOSSE::create();
    gazeTracker = GazeTracker(hogFaceDetector, shapePredictor, tracker);
}

void drawFaces(const Mat &frame, const std::vector<dlib::rectangle> &faceRects) {
    for (auto &faceRect: faceRects) {

        int x1 = (int) faceRect.left();
        int y1 = (int) faceRect.top();
        int x2 = (int) faceRect.right();
        int y2 = (int) faceRect.bottom();

        cv::rectangle(frame, Point(x1, y1), Point(x2, y2),
                      GREEN_COLOR, 1, 4);

    }
}

void drawLandmarks(const Mat &frame, const std::vector<dlib::full_object_detection> &facesLandmarks) {
    for (auto &landmarks: facesLandmarks) {
        for (int i = 0; i < landmarks.num_parts(); i++) {
            point p = landmarks.part(i);
            int x = (int) p.x();
            int y = (int) p.y();
            cv::circle(frame, Point(x, y), 2, RED_COLOR);
        }
    }
}
