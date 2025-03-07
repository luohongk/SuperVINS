/*
 * @Author: Hongkun Luo
 * @Date: 2024-11-05 21:16:57
 * @LastEditors: luohongk luohongkun@whu.edu.cn
 * @Description: 
 * 
 * Hongkun Luo
 */
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

#include "feature_tracker_dpl.h"
void FeatureTrackerDPL::initializeExtractorMatcher(int extractor_type_, string &extractor_weight_path, string &matcher_weight_path, float matcher_threshold = 0.5)
{
    extractor_type = extractor_type_;
    FeatureExtractorDPL = std::make_shared<Extractor_DPL>();
    FeatureExtractorDPL->initialize(extractor_weight_path, extractor_type_);

    FeatureMatcherDPL = std::make_shared<Matcher_DPL>();
    FeatureMatcherDPL->initialize(matcher_weight_path, extractor_type_, matcher_threshold);

    if (extractor_type_ == SUPERPOINT)
    {
        descriptor_size = SUPERPOINT_SIZE;
    }
    else if (extractor_type_ == DISK)
    {
        descriptor_size = DISK_SIZE;
    }
}

cv::Mat FeatureTrackerDPL::Extractor_PreProcess(const cv::Mat &Image, float &scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    // std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;

    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resize_img = ResizeImage(tempImage, IMAGE_SIZE_DPL, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(resize_img);
    // if (cfg.extractorType == "superpoint")
    //{
    // std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
    // resultImage = RGB2Grayscale(resultImage);
    //}
    // std::cout << "[INFO] Scale from " << temp_scale << " to " << scale << std::endl;

    return resultImage;
}

void FeatureTrackerDPL::goodFeaturesToTrack_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors, int max_num, double extractor_threshold, int radius, cv::Mat &mask)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);
    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {
        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        if (mask.at<uchar>(pt) == 255)
        {
            std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
            pts.push_back(pt);
            dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
        }
    }
}

void FeatureTrackerDPL::extract_features_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);

    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {

        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
        pts.push_back(pt);
        dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
    }
    // std::cout<<dplpts_descriptors[0].first<<endl;
    // std::cout<<dplpts_descriptors[0].second<<endl;
}

bool FeatureTrackerDPL::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void FeatureTrackerDPL::reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTrackerDPL::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTrackerDPL::FeatureTrackerDPL()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

void FeatureTrackerDPL::match_features_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    // 定义特征点与描述子大小
     // Define feature points and descriptor size
    int n_pre = prev_dplpts_descriptors_.size();
    int n_cur = cur_dplpts_descriptors_.size();
    // debug
    // cout << "prev_dples size = " << n_pre << "cur_dpls size=" << n_cur << endl;
    vector<cv::Point2f> prev_dplpts, cur_dplpts;
    prev_dplpts.reserve(n_pre);
    cur_dplpts.reserve(n_cur);
    float prev_descriptors[n_pre * descriptor_size];
    float cur_descriptors[n_cur * descriptor_size];

    for (int i = 0; i < n_pre; i++)
    {
        prev_dplpts.push_back(prev_dplpts_descriptors_[i].first);
        vector<float> desc = prev_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            prev_descriptors[idx] = desc_value;
            idx++;
        }
    }

    for (int i = 0; i < n_cur; i++)
    {
        cur_dplpts.push_back(cur_dplpts_descriptors_[i].first);
        vector<float> desc = cur_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            cur_descriptors[idx] = desc_value;
            idx++;
        }
    }

    // 深度学习特征匹配预处理
     // Deep learning feature matching preprocessing
    vector<cv::Point2f> prev_dplpts_normalized = FeatureMatcherDPL->pre_process(prev_dplpts, prev_img_.rows, prev_img_.cols);
    vector<cv::Point2f> cur_dplpts_normalized = FeatureMatcherDPL->pre_process(cur_dplpts, cur_img_.rows, cur_img_.cols);

    // 正式匹配特征点
    // Formal matching feature points
    vector<pair<int, int>> tem_matches;
    tem_matches = FeatureMatcherDPL->match_featurepoints(prev_dplpts_normalized, cur_dplpts_normalized, prev_descriptors, cur_descriptors);

    // std::cout<<"tem_matches size = "<<tem_matches.size()<<std::endl;

    // 假设有两个名为 keypoints1 和 keypoints2 的关键点向量，分别对应两个图像
    // std::vector<cv::KeyPoint> prev_dplpts_normalized, cur_dplpts_normalized;

    // Suppose there are two keypoint vectors named keypoints1 and keypoints2, corresponding to two images respectively.
    vector<cv::Point2f> points1, points2;
    // 假设从 matches 中提取出了对应的关键点
    // Assume that the corresponding key points are extracted from matches
    for (const auto &match : tem_matches)
    {
        cv::Point2f pt1 = prev_dplpts_normalized[match.first];
        cv::Point2f pt2 = cur_dplpts_normalized[match.second];

        // 将这些点存储在两个点向量中
        // Store these points in two point vectors
        points1.push_back(pt1);
        points2.push_back(pt2);
    }

    // RANSAC 参数，// RANSAC 阈值
    // RANSAC parameters, //RANSAC threshold
    // double ransacReprojThreshold = 0.05; 
    // cout<<"RANSAC threshold = "<<ransacReprojThreshold<<endl;

    // 使用 RANSAC 进行模型估计
    // Model estimation using RANSAC
    std::vector<uchar> inliersMask;

    // std::cout<<"start find fundamental matrix"<<std::endl;
    // std::cout<<"points1 size = "<<points1.size()<<std::endl;
    // std::cout<<"points2 size = "<<points2.size()<<std::endl;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, ransacReprojThreshold, 0.99, inliersMask);

    // std::cout<<"end find fundamental matrix"<<std::endl;

    // 获取内点
    // Get interior points
    // std::vector<cv::Point2f> inlierPrevPts, inlierCurPts;
    std::vector<pair<int, int>> inlierMatches;
    for (int i = 0; i < inliersMask.size(); ++i)
    {
        if (inliersMask[i])
        {
            result_matches.push_back(tem_matches[i]);
        }
    }
}

void FeatureTrackerDPL::match_with_predictions_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<cv::Point2f> &predict_pts_, vector<cv::Point2f> &cur_pts_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    int n_pre = prev_dplpts_descriptors_.size();
    int n_cur = cur_dplpts_descriptors_.size();
    vector<cv::Point2f> prev_dplpts, cur_dplpts;
    prev_dplpts.reserve(n_pre);
    cur_dplpts.reserve(n_cur);
    float prev_descriptors[n_pre * descriptor_size];
    float cur_descriptors[n_cur * descriptor_size];

    for (int i = 0; i < n_pre; i++)
    {
        prev_dplpts.push_back(prev_dplpts_descriptors_[i].first);
        vector<float> desc = prev_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            prev_descriptors[idx] = desc_value;
            idx++;
        }
    }

    for (int i = 0; i < n_cur; i++)
    {
        cur_dplpts.push_back(cur_dplpts_descriptors_[i].first);
        vector<float> desc = cur_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            cur_descriptors[idx] = desc_value;
            idx++;
        }
    }

    vector<cv::Point2f> prev_dplpts_normalized = FeatureMatcherDPL->pre_process(prev_dplpts, prev_img_.rows, prev_img_.cols);
    vector<cv::Point2f> cur_dplpts_normalized = FeatureMatcherDPL->pre_process(cur_dplpts, cur_img_.rows, cur_img_.cols);

    vector<pair<int, int>> matches = FeatureMatcherDPL->match_featurepoints(prev_dplpts_normalized, cur_dplpts_normalized, prev_descriptors, cur_descriptors);
    // RANSAC 参数
    // double ransacReprojThreshold = 0.06; // RANSAC 阈值
    // cout<<"RANSAC threshold = "<<ransacReprojThreshold<<endl;

    // 使用 RANSAC 进行模型估计
    std::vector<uchar> inliersMask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(prev_dplpts_normalized, cur_dplpts_normalized, cv::FM_RANSAC, ransacReprojThreshold, 0.99, inliersMask);

    // 获取内点
    std::vector<cv::Point2f> inlierPrevPts, inlierCurPts;
    std::vector<pair<int, int>> inlierMatches;
    for (int i = 0; i < inliersMask.size(); ++i)
    {
        if (inliersMask[i])
        {
            inlierPrevPts.push_back(prev_dplpts_normalized[i]);
            inlierCurPts.push_back(cur_dplpts_normalized[i]);
            result_matches.push_back(matches[i]);
        }
    }
}

/// @brief 设置特征提取掩码，并根据追踪次数排序
/// @brief sets the feature extraction mask and sorts according to the number of tracking
void FeatureTrackerDPL::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    // 将当前帧特征点按照（追踪次数，（特征点坐标，id））存储
    // Store the feature points of the current frame according to (number of tracking, (feature point
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // 按照追踪次数从大到小排序
    // Sort by tracking number from largest to smallest
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    // 将排序后的特征点存回去
    // Save the sorted feature points back
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            // 如果掩码内特征点像素位置是255，则在掩码周围设置一个半径为MIN_DIST的“禁止提取区域”,像素置为0
            // If the pixel position of the feature point in the mask is 255, set a "prohibited extraction area" with a radius of min dist around the mask, and set the pixel to 0
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); 
        }
    }
}

cv::Mat FeatureTrackerDPL::setMask_dpl(vector<cv::Point2f> &matched_points, int radius = MIN_DIST)
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    for (cv::Point2f &pt : matched_points)
    {
        if (mask.at<uchar>(pt) == 255)
        {
            cv::circle(mask, pt, radius, 0, -1); 
            // 如果掩码内特征点像素位置是255，则在掩码周围设置一个半径为MIN_DIST的“禁止提取区域”,像素置为0
            // If the pixel position of the feature point in the mask is 255, set a "prohibited extraction area" with a radius of min dist around the mask, and set the pixel to 0
        }
    }

    return mask.clone();
}

double FeatureTrackerDPL::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}



/// @brief 双目特征帧间追踪
/// @param _cur_time
/// @param _img
/// @param _img1
/// @return 图像特征

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTrackerDPL::trackImage_dpl(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    // 当前时间
    // current time
    cur_time = _cur_time;  

    // 当前左目图像   
    cur_img = _img;  

    // 图像行数 
    // Number of image lines         
    row = cur_img.rows;

    // 图像列数    
    // Number of image columns   
    col = cur_img.cols;

    // 当前右目图像 
    // Current right eye image 
    cv::Mat rightImg = _img1; 

    // 清空当前帧特征点
    // Clear feature points of current frame
    cur_pts.clear();
    cur_dplpts_descriptors.clear();

    // 开始计时
    cout<<"extract feature time"<<endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 使用深度学习进行特征点和描述子提取
    // use deep-learning extractor to extract feature points and descriptors
    extract_features_dpl(cur_img, cur_pts, cur_dplpts_descriptors);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start; // 计算持续时间

    // 输出时间到控制台
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // 打开文件进行保存（以追加模式打开）
    std::ofstream outFile("time_consumption/feature_extraction_matching.txt", std::ios::app);
    if (outFile.is_open()) {
        outFile << duration.count() << ","; // 写入执行时间
        outFile.close(); // 关闭文件
    } else {
        std::cerr << "Unable to open file" << std::endl; // 错误处理
    }

    cout<<"save success"<<endl;

    // 一些变量容器
    // some variables containers
    set<int> cur_matched_indices;
    vector<cv::Point2f> temp_prev_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_prev_dplpts_descriptors;
    vector<cv::Point2f> temp_cur_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_cur_dplpts_descriptors;
    vector<int> temp_ids;
    vector<int> temp_track_cnt;

    cout << cur_pts.size() << " points extracted in current frame originally" << endl;

    // 上一帧特征点不能为空，否则没法追踪
    // The feature points of the previous frame cannot be empty, otherwise they cannot be tracked.\

    ROS_INFO_STREAM("prev_pts.size() = " << prev_pts.size());
    
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<pair<int, int>> matches;

        std::cout<<"hasPrediction = "<<hasPrediction<<std::endl;
        if (hasPrediction)
        {
            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();

            //  检查图像是否有效
            // if (cur_img.empty()) {
            //     std::cerr << "无法读取图像。" << std::endl;
            // }
            // if(cv::countNonZero(cur_img) == 0){
            //     std::cerr << "图像全黑。" << std::endl;
            // }
            
            // 使用深度学习进行特征匹配
            // use deep-learning extractor to match feature points
            match_with_predictions_dpl(prev_img, cur_img, prev_dplpts_descriptors, cur_dplpts_descriptors, predict_pts, cur_pts, matches,ransacReprojThreshold);

            // ROS_INFO_STREAM("matches.size() = " << matches.size());
            
            // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start; // 计算持续时间

            // 输出时间到控制台
            // std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

            // 打开文件进行保存（以追加模式打开）
            std::ofstream outFile("time_consumption/feature_extraction_matching.txt", std::ios::app);
            if (outFile.is_open()) {
                outFile << duration.count() << std::endl;
                outFile.close();
            } else {
                std::cerr << "Unable to open file" << std::endl;
            }
        }
        else
        {
            // 检查图像是否有效
            // ROS_INFO_STREAM("cur_img.empty() = " << cur_img.empty());
            
            // if (cur_img.empty()) {
            //         std::cerr << "无法读取图像。" << std::endl;
            //     }

            // ROS_INFO_STREAM("cv::countNonZero(cur_img) = " << cv::countNonZero(cur_img));
            
            // if(cv::countNonZero(cur_img) == 0){
            //         std::cerr << "图像全黑。" << std::endl;
            //     }
            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();

            // 使用深度学习进行特征匹配
            // use deep-learning extractor to match feature points
            match_features_dpl(prev_img, cur_img, prev_dplpts_descriptors, cur_dplpts_descriptors, matches,ransacReprojThreshold);

            ROS_INFO_STREAM("matches.size() = " << matches.size());

            // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start; // 计算持续时间

            // 输出时间到控制台
            std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

            // 打开文件进行保存（以追加模式打开）
            std::ofstream outFile("duration.txt", std::ios::app);
            if (outFile.is_open()) {
                outFile << duration.count() <<std::endl; // 写入执行时间
                outFile.close(); // 关闭文件
            } else {
                std::cerr << "Unable to open file" << std::endl; // 错误处理
            }
        }

        // the number of matched points
        int n_matches = matches.size();
        // cout << n_matches <<" points matched in current frame" <<endl;

        // first, save the successfully matched points in the temporary vector
        for (int i = 0; i < n_matches; i++)
        {
            pair<int, int> match = matches[i];
            temp_prev_pts.push_back(prev_pts[match.first]);
            temp_prev_dplpts_descriptors.push_back(prev_dplpts_descriptors[match.first]);
            temp_cur_pts.push_back(cur_pts[match.second]);
            temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[match.second]);
            temp_ids.push_back(ids[match.first]);
            temp_track_cnt.push_back(track_cnt[match.first] + 1);
            // record the indices of matched points
            cur_matched_indices.insert(match.second);
        }
    }

    // to avoid point gathering, create a mask depending on matched points in the current image
    cv::Mat mask_for_dpl = setMask_dpl(temp_cur_pts, MIN_DIST);
    
    // add new feature points but filter the ones which are close to matched points via the mask
    for (int i = 0; i < cur_pts.size(); i++)
    {
        if (!cur_matched_indices.count(i))
        {
            if (mask_for_dpl.at<uchar>(cur_pts[i]) == 255) // TO DO: annote this line can alleviate the problem of system crashing
            {
                temp_cur_pts.push_back(cur_pts[i]);
                temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[i]);
                temp_track_cnt.push_back(1);
                temp_ids.push_back(n_id++);
            }
        }
    }

    prev_pts = temp_prev_pts;
    prev_dplpts_descriptors = temp_prev_dplpts_descriptors;
    cur_pts.swap(temp_cur_pts);
    cur_dplpts_descriptors.swap(temp_cur_dplpts_descriptors);
    track_cnt = temp_track_cnt;
    ids = temp_ids;

    // cout << cur_pts.size()<<" points reserved in current frame finally" <<endl;

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    if (!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        cur_dplpts_right_descriptors.clear();

        vector<cv::Point2f> temp_cur_right_pts;

        if (!cur_pts.empty())
        {
            extract_features_dpl(rightImg, cur_right_pts, cur_dplpts_right_descriptors);
            vector<pair<int, int>> right_matches;

            // double ransac_thresh = 0.5;
            match_features_dpl(cur_img, rightImg, cur_dplpts_descriptors, cur_dplpts_right_descriptors, right_matches,ransacReprojThreshold);
            int n_matches_right = right_matches.size();
            for (int i = 0; i < n_matches_right; i++)
            {
                pair<int, int> right_match = right_matches[i];
                temp_cur_right_pts.push_back(cur_right_pts[right_match.second]);
                ids_right.push_back(ids[right_match.first]);
            }
            cur_right_pts = temp_cur_right_pts;
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);                                              // 去畸变加反投影至归一化平面
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map); // 计算右目特征点在相机归一化平面上的速度
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    // 要显示追踪图片的话，需要画图
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // 基本上追踪完毕了，更新相关的变量
    prev_img = cur_img;               // 当前帧赋给上一帧
    prev_pts = cur_pts;               // 当前帧特征点赋给上一帧特征点
    prev_un_pts = cur_un_pts;         // 当前帧去畸变反投影特征点赋给上一帧去畸变反投影特征点
    prev_un_pts_map = cur_un_pts_map; // 上一帧去畸变反投影特征点的map=当前帧去畸变反投影特征点map：（特征点id,特征点坐标）
    prev_dplpts_descriptors = cur_dplpts_descriptors;
    prev_time = cur_time;  // 当前帧时间戳赋给上一帧时间戳
    hasPrediction = false; // 将特征点预测信号置0

    prevLeftPtsMap.clear(); // 上一帧左目特征点Map清空，下面再填充进当前帧的信息
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    ROS_INFO_STREAM("start save ");
    

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame; // 要准备返回特征帧了：key值为featureid，value值为map：key值为相机id，value值为7x1向量，分别存放[归一化平面x,归一化平面y,1,像素平面x,像素平面y,归一化平面速度x,归一化平面速度y]
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y, z;
        x = cur_un_pts[i].x; // 归一化平面x
        y = cur_un_pts[i].y; // 归一化平面y
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x; // 像素平面x
        p_v = cur_pts[i].y; // 像素平面y
        int camera_id = 0;  // 相机id，这里0是左目主相机
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x; // 归一化平面x方向速度
        velocity_y = pts_velocity[i].y; // 归一化平面y方向速度

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;      // 把数据存入向量
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity); // 形成当前帧的特征
    }
    // 右目是一样的流程
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1; // 其它和左目差不多，只有这个相机id是1，
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity); // 把右目数据推入
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTrackerDPL::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTrackerDPL::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTrackerDPL::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/// @brief 将像素平面特征点去畸变，并反投影至归一化相机系下
/// @param pts
/// @param cam
/// @return
vector<cv::Point2f> FeatureTrackerDPL::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);                                   // 针孔相机模型下，返回的b就是去畸变的归一化相机平面坐标[x,y,1]
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // 针孔相机模型下，b.z()是1.0，所以横纵坐标为归一化平面坐标
    }
    return un_pts;
}

/// @brief 计算特征点的速度，这个速度是指在特征点在相机归一化平面中的位移的速度
/// @param ids
/// @param pts 当前这一帧的特征点
/// @param cur_id_pts 名字叫cur_id_pts，但传进来的时候其实保留的是上一帧的特征点
/// @param prev_id_pts 传进来的时候保留的其实也是上一帧的特征点
/// @return
vector<cv::Point2f> FeatureTrackerDPL::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                   map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity; // 将要返回的特征点速度
    cur_id_pts.clear();               // 清空，下面放入当前帧的特征点信息
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    // 现在cur_id_pts里存的是当前帧特征点了，和prev_id_pts存的上一帧特征点计算速度
    if (!prev_id_pts.empty()) // 若上一帧特征点非空，则计算速度
    {
        double dt = cur_time - prev_time; // 时间间隔

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) // 能找到和上一帧对应的特征点，则计算速度
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0)); // 不能找到和上一帧对应的特征点，则速度为0
        }
    }
    else // 若上一帧特征点为空，则当前帧所有特征点速度初始化为0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTrackerDPL::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                  vector<int> &curLeftIds,
                                  vector<cv::Point2f> &curLeftPts,
                                  vector<cv::Point2f> &curRightPts,
                                  map<int, cv::Point2f> &prevLeftPtsMap)
{
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_BGR2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/// @brief 给FeatureTracker输入预测的空间点
/// @param predictPts
void FeatureTrackerDPL::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true; // 有预测特征点的信号置1
    predict_pts.clear();  // 把老的预测特征点清空
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict; // 设置一个存放路标点的map的迭代器
    // 遍历待追踪的特征点的id
    for (size_t i = 0; i < ids.size(); i++)
    {
        // printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);   // 找对应id的预测路标点
        if (itPredict != predictPts.end()) // 如果没找到
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);       // 将预测的相机系下的空间点投影至图像像素
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y())); // 存入预测特征点像素
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]); // 若是没有预测特征点，则利用历史特征点的位置（这其实就是不通过初始光流时的结果）
    }
}

void FeatureTrackerDPL::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTrackerDPL::getTrackImage()
{
    return imTrack;
}