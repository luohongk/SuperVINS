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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
// #include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
// #include "XFeat.h"

#include <chrono>
#include <iostream>
#include "omp.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#define MIN_LOOP_NUM 18

using namespace Eigen;
using namespace std;
using namespace DVision;

class BriefExtractor
{
public:
	virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
	BriefExtractor(const std::string &pattern_file);

	DVision::BRIEF m_brief;
};

class XFeat
{
public:
	XFeat(std::string &xfeatModelPath, std::string &matchingModelPath);
	int detectAndCompute(const cv::Mat &image, cv::Mat &mkpts, cv::Mat &feats, cv::Mat &sc);
	bool initOrtSession(const Ort::Env &env, Ort::Session &session, std::string &modelPath, int &gpuId);
	int matchStar(const cv::Mat &mkpts0, const cv::Mat &feats0, const cv::Mat &sc0, const cv::Mat &mkpts1, const cv::Mat &feats1, cv::Mat &matches, cv::Mat &batch_indexes);

	static cv::Mat warpCornersAndDrawMatches(const std::vector<cv::Point2f> &refPoints, const std::vector<cv::Point2f> &dstPoints,
											 const cv::Mat &img1, const cv::Mat &img2);

	const ORTCHAR_T *stringToOrtchar_t(std::string const &s);

	~XFeat();

	// gpu id
	int gpuId_ = 0;

	// onnxruntime
	Ort::Env env_{nullptr};
	Ort::Session xfeatSession_{nullptr};
	Ort::Session matchingSession_{nullptr};
	Ort::AllocatorWithDefaultOptions allocator;

	//
	std::vector<const char *> xfeatInputNames = {"images"};
	std::vector<const char *> xfeatOutputNames = {"mkpts", "feats", "sc"};
	std::vector<const char *> matchingInputNames = {"mkpts0", "feats0", "sc0", "mkpts1", "feats1"};
	std::vector<const char *> matchingOutputNames = {"matches", "batch_indexes"};

	bool initFinishedFlag_ = false;
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
			 vector<double> &_point_id, int _sequence);

	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
			 vector<double> &_point_id, int _sequence,cv::Mat descriptors);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame *old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();

	// new code
	void computeXFeatPoint();

	// void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
					  const std::vector<BRIEF::bitset> &descriptors_old,
					  const std::vector<cv::KeyPoint> &keypoints_old,
					  const std::vector<cv::KeyPoint> &keypoints_old_norm,
					  cv::Point2f &best_match,
					  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
						  std::vector<uchar> &status,
						  const std::vector<BRIEF::bitset> &descriptors_old,
						  const std::vector<cv::KeyPoint> &keypoints_old,
						  const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
								const std::vector<cv::Point2f> &matched_2d_old_norm,
								vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
				   const std::vector<cv::Point3f> &matched_3d,
				   std::vector<uchar> &status,
				   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();

	double time_stamp;
	int index;
	int local_index;
	Eigen::Vector3d vio_T_w_i;
	Eigen::Matrix3d vio_R_w_i;
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d;
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<cv::KeyPoint> keypoints;

	// new code
	cv::Mat XFeatKeypoints1000;
	cv::Mat XFeatKeypoints4800;

	vector<cv::KeyPoint> keypoints_norm;

	// new code
	cv::Mat XFeatKeypoints_norm1000;
	cv::Mat XFeatKeypoints_norm4800;

	vector<cv::KeyPoint> window_keypoints;

	vector<BRIEF::bitset> brief_descriptors;

	// new code
	cv::Mat XFeatDescriptors1000;
	cv::Mat XFeatDescriptors4800;

	// new code
	cv::Mat SuperPointDescriptors;

	vector<BRIEF::bitset> window_brief_descriptors;
	bool has_fast_point;
	int sequence;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1> loop_info;
};
