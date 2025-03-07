/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

// lightglue
#include "extractor_matcher_dpl.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

class FeatureTrackerDPL
{
public:
    FeatureTrackerDPL();
    // new codes: lightglue
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage_dpl(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
    void reduceVector(vector<int> &v, vector<uchar> status);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;

    int extractor_type = 0;
    int descriptor_size = 256;
    unsigned int IMAGE_SIZE_DPL=512; 
    void initializeExtractorMatcher(int extractor_type_, string &extractor_weight_path, string &matcher_weight_path, float matcher_threshold);
    std::shared_ptr<Extractor_DPL> FeatureExtractorDPL;
    std::shared_ptr<Matcher_DPL> FeatureMatcherDPL;

    vector<pair<cv::Point2f, vector<float>>> prev_dplpts_descriptors, cur_dplpts_descriptors, cur_dplpts_right_descriptors;

    // functions for replace extractor
    cv::Mat Extractor_PreProcess(const cv::Mat &srcImage, float &scale);
    void goodFeaturesToTrack_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors, int max_num, double extractor_threshold, int radius, cv::Mat &mask);
    void extract_features_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors);

    // functions for replace matcher
    void match_features_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold);
    void match_with_predictions_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<cv::Point2f> &predict_pts_, vector<cv::Point2f> &cur_pts_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold);

    cv::Mat setMask_dpl(vector<cv::Point2f> &matched_points, int radius);
};
