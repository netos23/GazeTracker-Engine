//
// Created by Nikita Morozov on 26.04.2022.
//

#ifndef EYETRACKER_GAZETRACKER_H
#define EYETRACKER_GAZETRACKER_H

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/tracking.hpp>
#include "opencv2/videoio.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv/cv_image.h"
#include <dlib/image_processing.h>
#include <iostream>

#define LEFT_EYE  0
#define RIGHT_EYE  1
#define BOTH_EYE  2
#define BLINKING_VALUE 12
#define ANY_EYE  3
#define LEFT_VERTICAL_EYE_POINTS  37,38, 41, 40
#define RIGHT_VERTICAL_EYE_POINTS  43,44, 47, 46
#define EYE_VERTICAL_POINTS_COUNT 4

class GazeTracker {
private:
    dlib::frontal_face_detector hogFaceDetector;
    dlib::shape_predictor shapePredictor;
    cv::Ptr<cv::Tracker> tracker;
    int epoch;


    std::vector<dlib::rectangle> facesRect;
    std::vector<dlib::full_object_detection> facesLandmarks;
    bool dirty;

    bool isBlinkingEye(
            const dlib::point& topLeft, const dlib::point& topRight,
            const dlib::point& bottomLeft, const dlib::point& bottomRight
    );

    void updateLandmark();

public:
    GazeTracker();

    GazeTracker(const dlib::frontal_face_detector &hogFaceDetector, dlib::shape_predictor shapePredictor,
                cv::Ptr<cv::Tracker> tracker);

    void update(const cv::Mat &frame);

    bool isBlinking(int eye, int person = 0);

    const std::vector<dlib::rectangle> &getFacesRect() const;

    void setFacesRect(const std::vector<dlib::rectangle> &facesRect);

    const std::vector<dlib::full_object_detection> &getFacesLandmarks() const;
};


#endif //EYETRACKER_GAZETRACKER_H
