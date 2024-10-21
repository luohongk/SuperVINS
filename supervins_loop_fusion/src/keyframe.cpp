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
	computeXFeatPoint();

	// std::cout <<"XFeat描述子的大小行为："<<XFeatDescriptors1000.rows<<"XFeat描述子的大小列为："<<XFeatDescriptors1000.cols<<std::endl;

	// std::cout<<"XFeatDescriptors1000第一行："<<XFeatDescriptors1000.row(0)<<std::endl;
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
	// computeXFeatPoint();
	SuperPointDescriptors=descriptors;

	// std::cout <<"XFeat描述子的大小行为："<<XFeatDescriptors1000.rows<<"XFeat描述子的大小列为："<<XFeatDescriptors1000.cols<<std::endl;

	// std::cout<<"XFeatDescriptors1000第一行："<<XFeatDescriptors1000.row(0)<<std::endl;
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

void KeyFrame::computeXFeatPoint()
{
	// ----------------------------------------------------------------------------
	// 变量定义与初始化
	std::string xfeatModelPath = "/home/lhk/catkin_ws/src/SuperVINS/supervins_loop_fusion/src/ThirdParty/DBoW3/xfeat/model/xfeat_dualscale.onnx";
	std::string matchingModelPath = "/home/lhk/catkin_ws/src/SuperVINS/supervins_loop_fusion/src/ThirdParty/DBoW3/xfeat/model/xfeat_matching.onnx";

	// 初始化xfeat对象
	// Init xfeat object
	XFeat xfeat(xfeatModelPath, matchingModelPath);

	cv::Mat sc0;
	cv::Mat keypoints;
	cv::Mat descriptors;
	cv::Mat temp_keypoints;
	cv::Mat temp_descriptors;

	// 特征点与描述符的提取
	xfeat.detectAndCompute(image, temp_keypoints, temp_descriptors, sc0);

	// ----------------------------------------------------------------------------
	// 接下来对特征点进行处理
	std::vector<cv::Mat> many_keypoints;
	cv::split(temp_keypoints, many_keypoints);

	// 使用 cv::vconcat() 将所有 cv::Mat 实例垂直拼接
	cv::vconcat(many_keypoints, keypoints);

	cv::Mat transposed_keypoints;
	cv::transpose(keypoints, transposed_keypoints);

	XFeatKeypoints4800 = transposed_keypoints.rowRange(0, 4800);
	XFeatKeypoints1000 = transposed_keypoints.rowRange(0, 1000);

	// ----------------------------------------------------------------------------
	// 接下来对特征描述子进行处理
	// cv::Mat temp_descriptors;
	std::vector<cv::Mat> many_descriptors;
	cv::split(temp_descriptors, many_descriptors);

	// 使用 cv::vconcat() 将所有 cv::Mat 实例垂直拼接
	cv::vconcat(many_descriptors, descriptors);

	cv::Mat transposed_mat;
	cv::transpose(descriptors, transposed_mat);

	XFeatDescriptors4800 = transposed_mat.rowRange(0, 4800);
	XFeatDescriptors1000 = transposed_mat.rowRange(0, 1000);
	// ----------------------------------------------------------------------------
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

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

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
	// printf("search by des\n");
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
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
			cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
			putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

			putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
			cv::vconcat(notation, loop_match_img, loop_match_img);

			/*
			ostringstream path;
			path <<  "/home/tony-ws1/raw_data/loop_image/"
					<< index << "-"
					<< old_kf->index << "-" << "3pnp_match.jpg";
			cv::imwrite( path.str().c_str(), loop_match_img);
			*/
			if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
			{
				/*
				cv::imshow("loop connection",loop_match_img);
				cv::waitKey(10);
				*/
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

// new code
// --------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------
XFeat::XFeat(std::string &xfeatModelPath, std::string &matchingModelPath)
{
	const ORTCHAR_T *ortXfeatModelPath = stringToOrtchar_t(xfeatModelPath);
	const ORTCHAR_T *ortMatchingModelPath = stringToOrtchar_t(matchingModelPath);

	env_ = Ort::Env{OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, "xfeat_demo"}; //  ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_FATAL

	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	std::cout << "All available accelerators:" << std::endl;
	for (int i = 0; i < availableProviders.size(); i++)
	{
		std::cout << "  " << i + 1 << ". " << availableProviders[i] << std::endl;
	}

	// init sessions
	initOrtSession(env_, xfeatSession_, xfeatModelPath, gpuId_);
	initOrtSession(env_, matchingSession_, matchingModelPath, gpuId_);
}

XFeat::~XFeat()
{
	env_.release();
	xfeatSession_.release();
	matchingSession_.release();
}

bool XFeat::initOrtSession(const Ort::Env &env, Ort::Session &session, std::string &modelPath, int &gpuId)
{
	const ORTCHAR_T *ortModelPath = stringToOrtchar_t(modelPath);

	bool sessionIsAvailable = false;
	/*
	if (sessionIsAvailable == false)
	{
		try
		{
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			// try Tensorrt
			OrtTensorRTProviderOptions trtOptions{};
			trtOptions.device_id = gpuId;
			trtOptions.trt_fp16_enable = 1;
			trtOptions.trt_engine_cache_enable = 1;
			trtOptions.trt_engine_cache_path = "./trt_engine_cache";


			trtOptions.trt_max_workspace_size = (size_t)4 * 1024 * 1024 * 1024;

			session_options.AppendExecutionProvider_TensorRT(trtOptions);

			session = Ort::Session(env, ortModelPath, session_options);

			sessionIsAvailable = true;
			std::cout << "Using accelerator: Tensorrt" << std::endl;
		}
		catch (Ort::Exception e)
		{
			std::cout << "Exception code: " << e.GetOrtErrorCode() << ", exception: " << e.what() << std::endl;
			std::cout << "Failed to init Tensorrt accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
		catch (...)
		{
			std::cout << "Failed to init Tensorrt accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
	}
	*/

	if (sessionIsAvailable == false)
	{
		try
		{
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			OrtCUDAProviderOptions cuda0ptions;
			cuda0ptions.device_id = gpuId;
			cuda0ptions.gpu_mem_limit = 4 << 30;

			session_options.AppendExecutionProvider_CUDA(cuda0ptions);

			session = Ort::Session(env, ortModelPath, session_options);

			sessionIsAvailable = true;
			std::cout << "Using accelerator: CUDA" << std::endl;
		}
		catch (Ort::Exception e)
		{
			std::cout << "Exception code: " << e.GetOrtErrorCode() << ", exception: " << e.what() << std::endl;
			std::cout << "Failed to init CUDA accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
		catch (...)
		{
			std::cout << "Failed to init CUDA accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
	}
	if (sessionIsAvailable == false)
	{
		try
		{
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			session = Ort::Session(env, ortModelPath, session_options);

			sessionIsAvailable = true;
			std::cout << "Using accelerator: CPU" << std::endl;
		}
		catch (Ort::Exception e)
		{
			std::cout << "Exception code: " << e.GetOrtErrorCode() << ", exception: " << e.what() << std::endl;
			std::cout << "Failed to init CPU accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
		catch (...)
		{
			std::cout << "Failed to init CPU accelerator." << std::endl;
			sessionIsAvailable = false;
		}
	}

	if (sessionIsAvailable == true)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		// Get input layers count
		size_t num_input_nodes = session.GetInputCount();

		// Get input layer type, shape, name
		for (int i = 0; i < num_input_nodes; i++)
		{

			// Name
			std::string input_name = std::string(session.GetInputNameAllocated(i, allocator).get());

			std::cout << "Input " << i << ": " << input_name << ", shape: (";

			// Type
			Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

			ONNXTensorElementDataType type = tensor_info.GetElementType();

			// Shape
			std::vector<int64_t> input_node_dims = tensor_info.GetShape();

			for (int j = 0; j < input_node_dims.size(); j++)
			{
				std::cout << input_node_dims[j];
				if (j == input_node_dims.size() - 1)
				{
					std::cout << ")" << std::endl;
				}
				else
				{
					std::cout << ", ";
				}
			}
		}

		// Get output layers count
		size_t num_output_nodes = session.GetOutputCount();

		// Get output layer type, shape, name
		for (int i = 0; i < num_output_nodes; i++)
		{
			// Name
			std::string output_name = std::string(session.GetOutputNameAllocated(i, allocator).get());
			std::cout << "Output " << i << ": " << output_name << ", shape: (";

			// type
			Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

			ONNXTensorElementDataType type = tensor_info.GetElementType();

			// shape
			std::vector<int64_t> output_node_dims = tensor_info.GetShape();
			for (int j = 0; j < output_node_dims.size(); j++)
			{
				std::cout << output_node_dims[j];
				if (j == output_node_dims.size() - 1)
				{
					std::cout << ")" << std::endl;
				}
				else
				{
					std::cout << ", ";
				}
			}
		}
	}
	else
	{
		std::cout << modelPath << " is invalid model." << std::endl;
	}

	return sessionIsAvailable;
}

int XFeat::detectAndCompute(const cv::Mat &image, cv::Mat &mkpts, cv::Mat &feats, cv::Mat &sc)
{
	// Pre process
	cv::Mat preProcessedImage = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
	int stride = preProcessedImage.rows * preProcessedImage.cols;
#pragma omp parallel for
	for (int i = 0; i < stride; i++) // HWC -> CHW, BGR -> RGB
	{
		*((float *)preProcessedImage.data + i) = (float)*(image.data + i * 3 + 2);
		*((float *)preProcessedImage.data + i + stride) = (float)*(image.data + i * 3 + 1);
		*((float *)preProcessedImage.data + i + stride * 2) = (float)*(image.data + i * 3);
	}

	// Create input tensor
	int64_t input_size = preProcessedImage.rows * preProcessedImage.cols * 3;
	std::vector<int64_t> input_node_dims = {1, 3, preProcessedImage.rows, preProcessedImage.cols};
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(preProcessedImage.data), input_size, input_node_dims.data(), input_node_dims.size());
	assert(input_tensor.IsTensor());

	// Run sessionn
	auto output_tensors =
		xfeatSession_.Run(Ort::RunOptions{nullptr}, xfeatInputNames.data(),
						  &input_tensor, xfeatInputNames.size(), xfeatOutputNames.data(), xfeatOutputNames.size());
	assert(output_tensors.size() == xfeatOutputNames.size() && output_tensors.front().IsTensor());

	// Get outputs
	auto mkptsShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int dim1 = static_cast<int>(mkptsShape[0]); // 1
	int dim2 = static_cast<int>(mkptsShape[1]); // 4800
	int dim3 = static_cast<int>(mkptsShape[2]); // 2
	float *mkptsDataPtr = output_tensors[0].GetTensorMutableData<float>();
	// To cv::Mat
	mkpts = cv::Mat(dim1, dim2, CV_32FC(dim3), mkptsDataPtr).clone();

	auto featsShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(featsShape[0]); // 1
	dim2 = static_cast<int>(featsShape[1]); // 4800
	dim3 = static_cast<int>(featsShape[2]); // 64
	float *featsDataPtr = output_tensors[1].GetTensorMutableData<float>();
	feats = cv::Mat(dim1, dim2, CV_32FC(dim3), featsDataPtr).clone();

	auto scShape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(scShape[0]); // 1
	dim2 = static_cast<int>(scShape[1]); // 4800
	float *scDataPtr = output_tensors[2].GetTensorMutableData<float>();
	sc = cv::Mat(dim1, dim2, CV_32F, scDataPtr).clone();

	return 0;
}

int XFeat::matchStar(const cv::Mat &mkpts0, const cv::Mat &feats0, const cv::Mat &sc0, const cv::Mat &mkpts1, const cv::Mat &feats1, cv::Mat &matches, cv::Mat &batch_indexes)
{
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	int64_t mkpts0_size = mkpts0.rows * mkpts0.cols * mkpts0.channels();
	std::vector<int64_t> mkpts0_dims = {mkpts0.rows, mkpts0.cols, mkpts0.channels()}; // 1x4800x2
	Ort::Value mkpts0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(mkpts0.data), mkpts0_size, mkpts0_dims.data(), mkpts0_dims.size());

	int64_t feats0_size = feats0.rows * feats0.cols * feats0.channels();
	std::vector<int64_t> feats0_dims = {feats0.rows, feats0.cols, feats0.channels()}; // 1x4800x64
	Ort::Value feats0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(feats0.data), feats0_size, feats0_dims.data(), feats0_dims.size());

	int64_t sc0_size = sc0.rows * sc0.cols;
	std::vector<int64_t> sc0_dims = {sc0.rows, sc0.cols}; // 1x4800
	Ort::Value sc0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(sc0.data), sc0_size, sc0_dims.data(), sc0_dims.size());

	int64_t mkpts1_size = mkpts1.rows * mkpts1.cols * mkpts1.channels();
	std::vector<int64_t> mkpts1_dims = {mkpts1.rows, mkpts1.cols, mkpts1.channels()}; // 1x4800x2
	Ort::Value mkpts1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(mkpts1.data), mkpts1_size, mkpts1_dims.data(), mkpts1_dims.size());

	int64_t feats1_size = feats1.rows * feats1.cols * feats1.channels();
	std::vector<int64_t> feats1_dims = {feats1.rows, feats1.cols, feats1.channels()}; // 1x4800x64
	Ort::Value feats1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)(feats1.data), feats1_size, feats1_dims.data(), feats1_dims.size());

	// Create input tensors
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(mkpts0_tensor));
	input_tensors.push_back(std::move(feats0_tensor));
	input_tensors.push_back(std::move(sc0_tensor));
	input_tensors.push_back(std::move(mkpts1_tensor));
	input_tensors.push_back(std::move(feats1_tensor));

	// Run session
	auto output_tensors =
		matchingSession_.Run(Ort::RunOptions{nullptr}, matchingInputNames.data(),
							 input_tensors.data(), matchingInputNames.size(), matchingOutputNames.data(), matchingOutputNames.size());

	std::cout << output_tensors.size() << std::endl;
	std::cout << xfeatOutputNames.size() << std::endl;

	// assert(output_tensors.size() == xfeatOutputNames.size() && output_tensors.front().IsTensor());
	assert(output_tensors.size() == matchingOutputNames.size() && output_tensors.front().IsTensor());

	// Get outputs
	auto matchesShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int dim1 = static_cast<int>(matchesShape[0]); // num
	int dim2 = static_cast<int>(matchesShape[1]); // 4
	// To cv::Mat
	float *matchesDataPtr = output_tensors[0].GetTensorMutableData<float>();
	matches = cv::Mat(dim1, dim2, CV_32F, matchesDataPtr).clone();

	auto batch_indexesShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(batch_indexesShape[0]); // num

	float *batch_indexesDataPtr = output_tensors[0].GetTensorMutableData<float>();
	batch_indexes = cv::Mat(dim1, 1, CV_32F, batch_indexesDataPtr).clone();

	return 0;
}

cv::Mat XFeat::warpCornersAndDrawMatches(const std::vector<cv::Point2f> &refPoints, const std::vector<cv::Point2f> &dstPoints, const cv::Mat &img1, const cv::Mat &img2)
{
	// Step 1: Calculate the Homography matrix and mask
	cv::Mat mask;
	cv::Mat H = cv::findHomography(refPoints, dstPoints, cv::RANSAC, 3.5, mask, 1000, 0.999);
	mask = mask.reshape(1, mask.total()); // Flatten the mask

	// Step 2: Get corners of the first image (img1)
	std::vector<cv::Point2f> cornersImg1 = {cv::Point2f(0, 0), cv::Point2f(img1.cols - 1, 0),
											cv::Point2f(img1.cols - 1, img1.rows - 1), cv::Point2f(0, img1.rows - 1)};
	std::vector<cv::Point2f> warpedCorners(4);

	// Step 3: Warp corners to the second image (img2) space
	cv::perspectiveTransform(cornersImg1, warpedCorners, H);

	// Step 4: Draw the warped corners in image2
	cv::Mat img2WithCorners = img2.clone();
	for (size_t i = 0; i < warpedCorners.size(); i++)
	{
		cv::line(img2WithCorners, warpedCorners[i], warpedCorners[(i + 1) % 4], cv::Scalar(0, 255, 0), 4);
	}

	// Step 5: Prepare keypoints and matches for drawMatches function
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch> matches;
	for (size_t i = 0; i < refPoints.size(); i++)
	{
		if (mask.at<uchar>(i))
		{ // Only consider inliers
			keypoints1.emplace_back(refPoints[i], 5);
			keypoints2.emplace_back(dstPoints[i], 5);
		}
	}
	for (size_t i = 0; i < keypoints1.size(); i++)
	{
		matches.emplace_back(i, i, 0);
	}

	// Draw inlier matches
	cv::Mat imgMatches;
	cv::drawMatches(img1, keypoints1, img2WithCorners, keypoints2, matches, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	return imgMatches;
}

// for onnx model path
const ORTCHAR_T *XFeat::stringToOrtchar_t(std::string const &s)
{
#ifdef _WIN32
	const char *CStr = s.c_str();
	size_t len = strlen(CStr) + 1;
	size_t converted = 0;
	wchar_t *WStr;
	WStr = (wchar_t *)malloc(len * sizeof(wchar_t));
	mbstowcs_s(&converted, WStr, len, CStr, _TRUNCATE);

	return WStr;
#else
	return s.c_str();
#endif // _WIN32
}
