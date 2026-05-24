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

#include "keyframe.h"
#include "matcher_dpl.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
				   vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if (!DEBUG_IMAGE)
		image.release();
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
				   vector<double> &_point_id, int _sequence,cv::Mat descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	SuperPointDescriptors=descriptors;
	if (!DEBUG_IMAGE)
		image.release();
}

// 新构造函数：带 SuperPoint 关键点坐标（用于 LightGlue 回环匹配）
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
				   vector<double> &_point_id, int _sequence, cv::Mat descriptors, vector<cv::Point2f> &_superpoint_keypoints)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	superpoint_keypoints = _superpoint_keypoints;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	SuperPointDescriptors = descriptors;
	if (!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
				   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
				   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	// vio_T_w_i = _vio_T_w_i;
	// vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

// 这个函数是用来计算当前关键帧的一个小window区域内的BRIEF特征描述子。
void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

// 这个函数是用来计算当前关键帧整张图像的BRIEF特征描述子。
void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if (1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for (int i = 0; i < (int)tmp_pts.size(); i++)
		{
			cv::KeyPoint key;
			key.pt = tmp_pts[i];
			keypoints.push_back(key);
		}
	}
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
	m_brief.compute(im, keys, descriptors);
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
							const std::vector<BRIEF::bitset> &descriptors_old,
							const std::vector<cv::KeyPoint> &keypoints_old,
							const std::vector<cv::KeyPoint> &keypoints_old_norm,
							cv::Point2f &best_match,
							cv::Point2f &best_match_norm)
{
	cv::Point2f best_pt;
	int bestDist = 128;
	int bestIndex = -1;
	for (int i = 0; i < (int)descriptors_old.size(); i++)
	{

		int dis = HammingDis(window_descriptor, descriptors_old[i]);
		if (dis < bestDist)
		{
			bestDist = dis;
			bestIndex = i;
		}
	}
	// printf("best dist %d", bestDist);
	if (bestIndex != -1 && bestDist < 80)
	{
		best_match = keypoints_old[bestIndex].pt;
		best_match_norm = keypoints_old_norm[bestIndex].pt;
		return true;
	}
	else
		return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
								std::vector<uchar> &status,
								const std::vector<BRIEF::bitset> &descriptors_old,
								const std::vector<cv::KeyPoint> &keypoints_old,
								const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
	for (int i = 0; i < (int)window_brief_descriptors.size(); i++)
	{
		cv::Point2f pt(0.f, 0.f);
		cv::Point2f pt_norm(0.f, 0.f);
		if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
			status.push_back(1);
		else
			status.push_back(0);
		matched_2d_old.push_back(pt);
		matched_2d_old_norm.push_back(pt_norm);
	}
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
									  const std::vector<cv::Point2f> &matched_2d_old_norm,
									  vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
	if (n >= 8)
	{
		vector<cv::Point2f> tmp_cur(n), tmp_old(n);
		for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
		{
			double FOCAL_LENGTH = 460.0;
			double tmp_x, tmp_y;
			tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
			tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

			tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
			tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
		}
		cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
	}
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
						 const std::vector<cv::Point3f> &matched_3d,
						 std::vector<uchar> &status,
						 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	// for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	// printf("match size %d \n", matched_3d.size());
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Matrix3d R_inital;
	Vector3d P_inital;
	Matrix3d R_w_c = origin_vio_R * qic;
	Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;
	TicToc t_pnp_ransac;

	if (CV_MAJOR_VERSION < 3)
		solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
	else
	{
		if (CV_MINOR_VERSION < 2)
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
		else
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
	}

	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
		status.push_back(0);

	for (int i = 0; i < inliers.rows; i++)
	{
		int n = inliers.at<int>(i);
		status[n] = 1;
	}

	cv::Rodrigues(rvec, r);
	Matrix3d R_pnp, R_w_c_old;
	cv::cv2eigen(r, R_pnp);
	R_w_c_old = R_pnp.transpose();
	Vector3d T_pnp, T_w_c_old;
	cv::cv2eigen(t, T_pnp);
	T_w_c_old = R_w_c_old * (-T_pnp);

	PnP_R_old = R_w_c_old * qic.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool KeyFrame::findConnection(KeyFrame *old_kf)
{
	TicToc tmp_t;
	// printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	bool use_lightglue = false; // 标记使用的匹配方法 (用于可视化标注)
	int lightglue_raw_matches = 0; // LightGlue 原始匹配数 (包括无3D对应的)
	vector<cv::Point2f> all_lg_cur_pts, all_lg_old_pts; // 全部 LightGlue 匹配 (用于可视化)

	TicToc t_match;
#if 0
		if (DEBUG_IMAGE)
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif

	// ============================================================
	// LightGlue matching (SuperPoint descriptors)
	// If LightGlue matcher is available and both frames have SuperPoint data,
	// use LightGlue; otherwise fall back to BRIEF matching.
	// ============================================================
	if (g_loop_matcher != nullptr &&
	    !superpoint_keypoints.empty() && !old_kf->superpoint_keypoints.empty() &&
	    !SuperPointDescriptors.empty() && !old_kf->SuperPointDescriptors.empty())
	{
		// Step 1: Normalize keypoints for LightGlue input
		vector<cv::Point2f> cur_kpts_norm = g_loop_matcher->pre_process(superpoint_keypoints, ROW, COL);
		vector<cv::Point2f> old_kpts_norm = g_loop_matcher->pre_process(old_kf->superpoint_keypoints, ROW, COL);

		// Step 2: Run LightGlue matching
		auto lg_matches = g_loop_matcher->match_featurepoints(
		    cur_kpts_norm, old_kpts_norm,
		    (float *)SuperPointDescriptors.data,
		    (float *)old_kf->SuperPointDescriptors.data);

		use_lightglue = true;
		lightglue_raw_matches = (int)lg_matches.size();

		// Step 3: Build 2D-3D correspondences
		// For each LightGlue match, find the nearest VIO point in current frame
		// to establish 3D position for PnP
		const float NEAREST_PIXEL_THRESH = 15.0f; // pixel distance threshold

		for (auto &match : lg_matches)
		{
			cv::Point2f cur_sp_pt = superpoint_keypoints[match.first];
			cv::Point2f old_sp_pt = old_kf->superpoint_keypoints[match.second];

			// 保存所有 LightGlue 匹配用于可视化
			// Save all LightGlue matches for visualization
			all_lg_cur_pts.push_back(cur_sp_pt);
			all_lg_old_pts.push_back(old_sp_pt);

			// Find nearest VIO tracked point to this SuperPoint keypoint in current frame
			int nearest_vio_idx = -1;
			float min_dist = NEAREST_PIXEL_THRESH;
			for (int i = 0; i < (int)point_2d_uv.size(); i++)
			{
				float dx = cur_sp_pt.x - point_2d_uv[i].x;
				float dy = cur_sp_pt.y - point_2d_uv[i].y;
				float dist = sqrt(dx * dx + dy * dy);
				if (dist < min_dist)
				{
					min_dist = dist;
					nearest_vio_idx = i;
				}
			}

			if (nearest_vio_idx >= 0)
			{
				matched_3d.push_back(point_3d[nearest_vio_idx]);
				matched_2d_cur.push_back(point_2d_uv[nearest_vio_idx]);
				matched_2d_cur_norm.push_back(point_2d_norm[nearest_vio_idx]);
				matched_id.push_back(point_id[nearest_vio_idx]);

				// Old frame 2D point
				matched_2d_old.push_back(old_sp_pt);

				// Undistort old point to get normalized coordinates
				Eigen::Vector3d tmp_p;
				m_camera->liftProjective(Eigen::Vector2d(old_sp_pt.x, old_sp_pt.y), tmp_p);
				matched_2d_old_norm.push_back(cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z()));
			}
		}
		printf("[LightGlue Loop] raw_matches=%d, with_3D=%d (cur_sp=%d, old_sp=%d, vio=%d)\n",
		       lightglue_raw_matches, (int)matched_2d_cur.size(),
		       (int)superpoint_keypoints.size(), (int)old_kf->superpoint_keypoints.size(),
		       (int)point_2d_uv.size());
	}
	else
	{
		// Fallback: use BRIEF descriptor matching (original logic)
		matched_3d = point_3d;
		matched_2d_cur = point_2d_uv;
		matched_2d_cur_norm = point_2d_norm;
		matched_id = point_id;

		searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_id, status);
	}
	// printf("search by des finish\n");

#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
	status.clear();
/*
FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
reduceVector(matched_2d_cur, status);
reduceVector(matched_2d_old, status);
reduceVector(matched_2d_cur_norm, status);
reduceVector(matched_2d_old_norm, status);
reduceVector(matched_3d, status);
reduceVector(matched_id, status);
*/
#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_id, status);
#if 1
		if (DEBUG_IMAGE)
		{
			int gap = 10;
			cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			cv::hconcat(image, gap_image, gap_image);
			cv::hconcat(gap_image, old_img, gray_img);
			cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);

			if (use_lightglue)
			{
				// ---- LightGlue 可视化 ----
				// 1) 先画所有 LightGlue 原始匹配 (薄线, 浅青色) 展示匹配密度
				// Draw ALL LightGlue matches (thin lines, light cyan) to show matching density
				for (int i = 0; i < (int)all_lg_cur_pts.size(); i++)
				{
					cv::Point2f cur_pt = all_lg_cur_pts[i];
					cv::Point2f old_pt = all_lg_old_pts[i];
					old_pt.x += (COL + gap);
					cv::circle(loop_match_img, cur_pt, 3, cv::Scalar(200, 200, 0), 1);
					cv::circle(loop_match_img, old_pt, 3, cv::Scalar(200, 200, 0), 1);
					cv::line(loop_match_img, cur_pt, old_pt, cv::Scalar(180, 180, 0), 1, 8, 0);
				}

				// 2) 再画 PnP inlier 匹配 (粗线, 亮黄色) 高亮有效约束
				// Draw PnP inliers (thick lines, bright yellow) to highlight valid constraints
				for (int i = 0; i < (int)matched_2d_cur.size(); i++)
				{
					cv::Point2f cur_pt = matched_2d_cur[i];
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap);
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 255), -1);
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 255), -1);
					cv::line(loop_match_img, cur_pt, old_pt, cv::Scalar(0, 255, 255), 2, 8, 0);
				}
			}
			else
			{
				// ---- BRIEF 可视化 (保持原逻辑) ----
				for (int i = 0; i < (int)matched_2d_cur.size(); i++)
				{
					cv::Point2f cur_pt = matched_2d_cur[i];
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
				}
				for (int i = 0; i < (int)matched_2d_old.size(); i++)
				{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap);
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
				}
				for (int i = 0; i < (int)matched_2d_cur.size(); i++)
				{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap);
					cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
				}
			}

			// 标注信息栏: 方法类型 + 帧号 + 匹配统计
			// Annotation bar: method type + frame info + match statistics
			cv::Mat notation(80, COL + gap + COL, CV_8UC3, cv::Scalar(40, 40, 40));

			// 第一行: 匹配方法 + 统计
			string method_str = use_lightglue ? "[LightGlue]" : "[BRIEF]";
			string match_info;
			if (use_lightglue)
				match_info = method_str + " all:" + to_string(lightglue_raw_matches)
				           + " pnp_inlier:" + to_string((int)matched_2d_cur.size())
				           + " score:" + to_string(g_loop_matcher->last_avg_match_score).substr(0, 4);
			else
				match_info = method_str + " PnP inliers: " + to_string((int)matched_2d_cur.size());
			cv::Scalar text_color = use_lightglue ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
			putText(notation, match_info, cv::Point2f(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2);

			// 第二行: 帧号信息
			string frame_info = "cur:" + to_string(index) + "(seq" + to_string(sequence) + ")"
			                  + "  loop:" + to_string(old_kf->index) + "(seq" + to_string(old_kf->sequence) + ")";
			putText(notation, frame_info, cv::Point2f(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

			// 第三行 (仅 LightGlue): 特征点数量
			if (use_lightglue)
			{
				string kpt_info = "SP_cur:" + to_string((int)superpoint_keypoints.size())
				               + " SP_old:" + to_string((int)old_kf->superpoint_keypoints.size())
				               + " VIO:" + to_string((int)point_2d_uv.size());
				putText(notation, kpt_info, cv::Point2f(10, 73), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 150, 150), 1);
			}

			cv::vconcat(notation, loop_match_img, loop_match_img);

			if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
			{
				cv::Mat thumbimage;
				cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
				msg->header.stamp = ros::Time(time_stamp);
				pub_match_img.publish(msg);
			}
		}
#endif
	}

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		relative_q = PnP_R_old.transpose() * origin_vio_R;
		relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
		// printf("PNP relative\n");
		// cout << "pnp relative_t " << relative_t.transpose() << endl;
		// cout << "pnp relative_yaw " << relative_yaw << endl;
		if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
		{

			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
				relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
				relative_yaw;
			// cout << "pnp relative_t " << relative_t.transpose() << endl;
			// cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
			return true;
		}
	}
	// printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
	BRIEF::bitset xor_of_bitset = a ^ b;
	int dis = xor_of_bitset.count();
	return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = vio_T_w_i;
	_R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = T_w_i;
	_R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
	return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
	return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
	return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		// printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
	// The DVision::BRIEF extractor computes a random pattern by default when
	// the object is created.
	// We load the pattern that we used to build the vocabulary, to make
	// the descriptors compatible with the predefined vocabulary

	// loads the pattern
	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
		throw string("Could not open file ") + pattern_file;

	vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}
