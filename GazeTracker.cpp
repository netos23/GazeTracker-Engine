//
// Created by Nikita Morozov on 26.04.2022.
//

#include "GazeTracker.h"

#include <utility>

#define IS_BLINKING_LEFT_EYE(landmark) isBlinkingEye(landmark.part(37),landmark.part(38), \
landmark.part(41), landmark.part(40))
#define IS_BLINKING_RIGHT_EYE(landmark) isBlinkingEye(landmark.part(43),landmark.part(44), \
landmark.part(47), landmark.part(46))
#define MIDPOINT(left, right) dlib::point((left.x()+right.x())/2,(left.y()+right.y())/2)
#define POW2(x) ((x)*(x))
#define DISTANCE(from, to) sqrt(POW2(from.x()-to.x())+POW2(from.y()-to.y()))

cv::Mat getSubFrameFromPoints(const cv::Mat &src, const std::vector<dlib::point> &points, std::string id);

inline std::vector<dlib::point> getLeftEyePoints(const dlib::full_object_detection &faceLandmark) {
    std::vector<dlib::point> points;
    points.push_back(faceLandmark.part(36));
    points.push_back(faceLandmark.part(37));
    points.push_back(faceLandmark.part(38));
    points.push_back(faceLandmark.part(39));
    points.push_back(faceLandmark.part(40));
    points.push_back(faceLandmark.part(41));
    return points;
}


inline std::vector<dlib::point> getRightEyePoints(const dlib::full_object_detection &faceLandmark) {
    std::vector<dlib::point> points;
    points.push_back(faceLandmark.part(42));
    points.push_back(faceLandmark.part(43));
    points.push_back(faceLandmark.part(44));
    points.push_back(faceLandmark.part(45));
    points.push_back(faceLandmark.part(46));
    points.push_back(faceLandmark.part(47));
    return points;
}

GazeTracker::GazeTracker() = default;

void GazeTracker::setFacesRect(const std::vector<dlib::rectangle> &facesRect) {
    GazeTracker::facesRect = facesRect;
}

const std::vector<dlib::rectangle> &GazeTracker::getFacesRect() const {
    return facesRect;
}


void GazeTracker::update(const cv::Mat &frame) {
    cv::Mat greyFrame;
    cv::cvtColor(frame, greyFrame, CV_BGR2GRAY);
    dlib::cv_image<dlib::bgr_pixel> tempFrame(frame);
    if (dirty) {
        setFacesRect(hogFaceDetector(tempFrame));

        for (auto &faceRect: facesRect) {
            auto x = double(faceRect.left());
            auto y = double(faceRect.top());
            auto width = double(faceRect.width());
            auto height = double(faceRect.height());
            cv::Rect2d boundingRect(x, y, width, height);

            tracker.release();
            // 30-40 fps
//            tracker = cv::TrackerBoosting::create();
            // >100 fps
//            tracker = cv::TrackerMOSSE::create();
            tracker = cv::TrackerMedianFlow::create();

            tracker->init(frame, boundingRect);

            const cv::Point_<double> &tl = boundingRect.tl();
            const cv::Point_<double> &br = boundingRect.br();
            faceRect = dlib::rectangle(long(tl.x), long(tl.y), long(br.x), long(br.y));
            dirty = false;
        }


    }
    facesLandmarks.clear();
    for (auto &faceRect: facesRect) {

        auto x = double(faceRect.left());
        auto y = double(faceRect.top());
        auto width = double(faceRect.width());
        auto height = double(faceRect.height());
        cv::Rect2d boundingRect(x, y, width, height);

        dirty = !tracker->update(frame, boundingRect);

        if (dirty) {
            return;
        }
        epoch++;
        if (epoch > 25) {
            dirty = true;
            epoch = 1;
        }

        const cv::Point_<double> &tl = boundingRect.tl();
        const cv::Point_<double> &br = boundingRect.br();
        faceRect = dlib::rectangle(long(tl.x), long(tl.y), long(br.x), long(br.y));

        dlib::full_object_detection landmarks = shapePredictor(tempFrame, faceRect);
        facesLandmarks.push_back(landmarks);
    }
}

bool GazeTracker::isBlinking(int eye, int person) {
    if (facesLandmarks.size() <= person) {
        return false;
    }
    dlib::full_object_detection faceLandmark = facesLandmarks[person];

    switch (eye) {
        case LEFT_EYE:
            return IS_BLINKING_LEFT_EYE(faceLandmark);
        case RIGHT_EYE:
            return IS_BLINKING_RIGHT_EYE(faceLandmark);
        case BOTH_EYE:
            return IS_BLINKING_LEFT_EYE(faceLandmark) && IS_BLINKING_RIGHT_EYE(faceLandmark);
        default:
            return IS_BLINKING_LEFT_EYE(faceLandmark) || IS_BLINKING_RIGHT_EYE(faceLandmark);
    }
}

bool GazeTracker::isBlinkingEye(
        const dlib::point &topLeft, const dlib::point &topRight,
        const dlib::point &bottomLeft, const dlib::point &bottomRight
) {
    dlib::point top = MIDPOINT(topLeft, topRight);
    dlib::point bottom = MIDPOINT(bottomLeft, bottomRight);
    double dist = DISTANCE(top, bottom);

    if (dist <= BLINKING_VALUE) {
        return true;
    }
    return false;
}


cv::Mat getSubFrameFromPoints(const cv::Mat &src, const std::vector<dlib::point> &points, std::string id) {
    int left = src.cols, right = 0, top = src.rows, bottom = 0;
    for (auto p: points) {
        left = MIN(left, p.x());
        right = MAX(right, p.x());
        top = MIN(top, p.y());
        bottom = MAX(bottom, p.y());
    }

    printf("%d %d %d %d\n", left, right, top, bottom);
    const cv::Mat &mat = src.colRange(left, right)
            .rowRange(top, bottom);

    cv::imshow("test" + id, mat);

    return mat;
}


void GazeTracker::updateLandmark() {

}

const std::vector<dlib::full_object_detection> &GazeTracker::getFacesLandmarks() const {
    return facesLandmarks;
}

GazeTracker::GazeTracker(const dlib::frontal_face_detector &hogFaceDetector,
                         dlib::shape_predictor shapePredictor, cv::Ptr<cv::Tracker> tracker)
        : hogFaceDetector(hogFaceDetector), shapePredictor(std::move(shapePredictor)), tracker(std::move(tracker)),
          dirty(true), epoch(0) {}




