//
// Created by Nikita Morozov on 04.05.2022.
//

#ifndef EYETRACKER_TRACKERS_H
#define EYETRACKER_TRACKERS_H

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/tracking.hpp>
#include "opencv2/videoio.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv/cv_image.h"
#include <dlib/image_processing.h>
#include <iostream>

namespace GazeTracker {

    typedef cv::Ptr<cv::Tracker> trackerFactory();

    class Tracker {
    public:
        virtual bool hasData() = 0;
    };

    class IFrameTracker : public Tracker {
    public:
        virtual void init(const cv::Mat &frame) = 0;

        virtual void update(const cv::Mat &frame) = 0;


    };

    class ILandmarkTracker : public Tracker {
    public:
        virtual void init(const dlib::full_object_detection &landmark) = 0;

        virtual void update(const dlib::full_object_detection &landmark) = 0;

    };

    class IFaceAndLandmarkTracker : public Tracker {
    public:
        virtual void init(const cv::Mat &frame, const dlib::full_object_detection &landmark) = 0;

        virtual void update(const cv::Mat &frame, const dlib::full_object_detection &landmark) = 0;
    };

    class FaceTracker : IFrameTracker {
    public:
    };

}


#endif //EYETRACKER_TRACKERS_H
