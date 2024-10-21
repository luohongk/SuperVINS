/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    // new codes: initialize deep learning extractor and matcher
    //  string extractorDPL_path = "/home/lhk/catkin_ws/src/VINS-Fusion-LightGlue/vins_estimator/weights_dpl/superpoint.onnx";
    //  string matcherDPL_path = "/home/lhk/catkin_ws/src/VINS-Fusion-LightGlue/vins_estimator/weights_dpl/superpoint_lightglue_fused_cpu.onnx";
    featureTracker.initializeExtractorMatcher(0, extractor_weight_global_path, matcher_weight_global_path, MATCHER_THRESHOLD); // initialize deep-learning based extractor and matcher

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

// 输入
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    // 输入图像计数加1
    inputImageCnt++;
    // 准备好特征帧来接收特征追踪器返回的结果
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    // 右目图像缺失的话，就只追踪左目图像
    if (_img1.empty())
    {
        // featureFrame = featureTracker.trackImage(t, _img);
        featureFrame = featureTracker.trackImage_dpl(t, _img); // track image with deeplearning methods

        // 将描述子保存成为cv::Mat
        vector<pair<cv::Point2f, vector<float>>> kptAndDescriptors = featureTracker.cur_dplpts_descriptors;

        std::cout << std::fixed << std::setprecision(9) << t << std::endl;
        // 先初始化描述子，cv::Mat形式
        cv::Mat descriptors(kptAndDescriptors.size(), 256, CV_32FC1);

        for (int i = 0; i < kptAndDescriptors.size(); i++)
        {
            for (int j = 0; j < 256; j++)
            {
                descriptors.at<float>(i, j) = kptAndDescriptors[i].second[j];
            }
        }
        
        // cout << descriptors.rows << endl;

        pair<double, cv::Mat> superPointDescriptors = make_pair(t, descriptors);

        // 将描述子推送到缓冲区
        mSuperPointDescriptors.lock();
        SuperPointDescriptorsBuf.push(make_pair(t, descriptors));
        mSuperPointDescriptors.unlock();
    }

    else
    {
        featureFrame = featureTracker.trackImage_dpl(t, _img, _img1); // track image with deeplearning methods

        // 如果是双目的话说，也是一样的处理，把数据推送到描述符缓冲区
        // 将描述子保存成为cv::Mat
        vector<pair<cv::Point2f, vector<float>>> kptAndDescriptors = featureTracker.cur_dplpts_descriptors;

        // 先初始化描述子，cv::Mat形式
        cv::Mat descriptors(kptAndDescriptors.size(), 256, CV_32FC1);

        for (int i = 0; i < kptAndDescriptors.size(); i++)
        {
            for (int j = 0; j < 256; j++)
            {
                descriptors.at<float>(i, j) = kptAndDescriptors[i].second[j];
            }
        }

        // 将描述子推送到缓冲区
        mSuperPointDescriptors.lock();
        SuperPointDescriptorsBuf.push(make_pair(t, descriptors));
        mSuperPointDescriptors.unlock();
    }

    if (SHOW_TRACK)
    {
        // featureTracker.trackImage里把特征点追踪效果图画好了，若要发布的话就发布
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    // 如果开启了多线程模式，processMeasurements()函数已经在一个独立的线程中运行了，所以只需要将新获取的特征追踪结果推入featureBuf即可
    if (MULTIPLE_THREAD)
    {
        if (inputImageCnt % 2 == 0) // XXX 估计器插入图像数目是偶数时才插入新图像帧，插入的是[时间戳,特征帧]，后面会用到
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame)); // 插入的是[时间戳,特征帧]，后面会用到
        mBuf.unlock();
        TicToc processTime;
        processMeasurements(); // 没开启多线程模式就要手动运行processMeasurements()
        // printf("process time: %f\n", processTime.toc());
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

/// @brief 获取t0和t1之间的IMU数据
/// @param t0 //上一帧时间戳
/// @param t1 //当前帧时间戳
/// @param accVector
/// @param gyrVector
/// @return
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if (t1 <= accBuf.back().first) // 如果当前帧时间戳t1小于等于最新的IMU数据时间戳
    {
        while (accBuf.front().first <= t0) // 最老的IMU数据小于上一帧时间戳时，就弹出这数据，因为这数据没法用于帧间预积分
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        // 上一个while结束以后，buffer里的数据就都是前一帧t0之后的了
        while (accBuf.front().first < t1) // 当最老的IMU数据小于当前帧时间戳，就把数据推入accVector、gyrVector
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front()); // 多推了一个数据进来，等于或略微大于t1
        gyrVector.push_back(gyrBuf.front());
    }
    else // 要是最新的数据都小于当前帧时间戳，说明可能还有可能有用的数据没输入
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/// @brief 判断时间戳t之后是否有可用IMU数据
/// @param t
/// @return
bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first) // 加速度计buffer不为空，最新的imu数据时间戳大于t（队列是先进先出的，.front是最老的，.back是新的），则有可用imu数据
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        // std_msgs::Header des_header;
        // des_header.frame_id = "world";
        // // 发布每一帧特征描述符的数据
        // pubSuperPointDescriptors(*this, des_header);

        // printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature; // 特征，double是时间戳，后面那个map和feature_tracker.track_image返回的featureframe一样，意思是这个时间戳下的所有特征
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;                     // double是时间戳
        if (!featureBuf.empty())
        {
            feature = featureBuf.front(); // 因为每追踪完一帧图像就把featureframe推入先进先出的buffer，所以要取最老的feature出来
            curTime = feature.first + td; // 当前时刻由特征（其实就是图像时间戳）加上一个固定延迟
            while (1)
            {
                if ((!USE_IMU || IMUAvailable(feature.first + td))) // 检查是否使用IMU，并检查是否有可用IMU数据
                    break;
                else // 如果没有就等待IMU数据
                {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD) // 非多线程时直接返回，等待IMU数据到来以后再接下去运行；如果多线程，则接着往下运行
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if (USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector); // 获取上一帧和当前帧之间的IMU数据

            featureBuf.pop(); // 弹出最老的特征，因为已经赋予feature这个变量了
            mBuf.unlock();

            if (USE_IMU)
            {
                if (!initFirstPoseFlag) // 要是没有初始化位姿，首先要进行IMU位姿初始化，将初始位姿与重力方向对齐
                    initFirstIMUPose(accVector);
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    /*                                                                                                   --------dt----------
                                                                                                                        |                    |
                    IMU数据之间的时间间隔，它的结构是：prevTime---dt--->imu[0]---dt--->imu[1]---dt--->imu[2]---...---imu[i-1]|---dt--->curTime-----imu[i]*/
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prevTime; // 因为accVector[i]肯定比prevTime大，所以第一个数据历元的时间间隔是第一个到上一帧之间的时间差异
                    else if (i == accVector.size() - 1)     // 如果accVector[i]中除最后一个历元以外都小于curTime，所以最倒数第二个历元的时间间隔是当前帧时间戳减去倒数第二个历元时间戳
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // 机械编排+预积分
                }
            }
            mProcess.lock();                             // 上锁，防止其它线程改变内参、状态量等等
            processImage(feature.second, feature.first); // 入口函数，状态估计都在这函数里面开始进行了
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);

            pubSuperPointDescriptors(*this, header);
            mProcess.unlock();
        }

        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/// @brief 初始化第一个IMU位姿
/// @param accVector
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;
    Eigen::Vector3d averAcc(0, 0, 0);
    // 计算平均加速度
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    // 计算平均加速度方向和[0,0,1]垂直方向的旋转关系
    Matrix3d R0 = Utility::g2R(averAcc);
    // 从R0中解出偏航角
    double yaw = Utility::R2ypr(R0).x();
    // 仅修正偏航角方向的旋转，将初始姿态与重力方向对齐
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl
         << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/// @brief 进行预积分，并用机械编排预测状态量
/// @param t imu数据的时间戳
/// @param dt imu数据时间戳和上一个历元时间戳之间的间隔
/// @param linear_acceleration t时刻imu数据的加计数据
/// @param angular_velocity t时刻imu数据的角速度数据
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 如果初始IMU数据还没赋值，则赋予上一个IMU数据变量当前这个IMU数据
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 如果第frame_count个预积分是空的，则要新建一个预积分量
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 如果当前帧不是初始第一帧，则要进行预积分
    if (frame_count != 0)
    {
        // 第frame_count个预积分量计算
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        // 临时预积分量计算
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // 第frame_count帧和上一帧之间的IMU数据存储
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        // 机械编排预测第frame_count帧的位姿和速度
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    // 将上一个IMU数据变量赋予当前IMU数据
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/// @brief 处理图像帧，状态估计
/// @param image
/// @param header
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // 检查特征的视差来判断边缘化情况（判断是否为加入滑窗的关键帧），同时这个函数也会将特征推入feature_manager
    {
        marginalization_flag = MARGIN_OLD; // 要是当前帧特征质量好，就把最老的关键帧边缘化掉
        // printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW; // 要是当前帧特征质量不怎么地，就边缘化新的帧
        // printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);                                                        // 构建当前图像帧
    imageframe.pre_integration = tmp_pre_integration;                                            // 赋予当前图像帧的预积分量临时预积分量
    all_image_frame.insert(make_pair(header, imageframe));                                       // 插入到所有图像帧库里
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // 上一个临时预积分已经完成了它的使命，马上要去用于当前帧状态估计了，所以现在要新建一个为下一帧服务

    if (ESTIMATE_EXTRINSIC == 2) // 如果要在线标定外参，就要找当前帧和上一帧之间的特征关联，结合预积分量进行外参优化
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 要是系统没初始化，需要先将系统初始化
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        // 单目视觉惯性初始化
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE) // 凑够一个滑窗的图像帧才能进行初始化
            {
                bool result = false;
                // 如果不需要在线标定外参，而且当前帧时间戳与初始时间戳大于0.1s，则可以进行sfm
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {

                    result = initialStructure();
                    initial_timestamp = header;
                }
                if (result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        // 双目视觉惯性初始化
        if (STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }
    else
    {
        TicToc t_solve;
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (!MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}

/// @brief 初始化结构，sfm出一些初始姿态、位置和点的位置
/// @return
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    // 先检查IMU可观性，实际上就是检查IMU是否有充分运动（激励）但这段代码算了半天最后也没起作用，要不咱直接看下面sfm的代码吧
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 遍历现有的所有帧
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;            // 取出每帧预积分的时间间隔
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 用速度变化量除以时间间隔得到每帧的平均速度变化量
            sum_g += tmp_g;                                                  // 将所有帧的平均速度变化值累加
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 算所有帧的平均速度变化量
        double var = 0;                                           // 新建一个变量用来保存方差
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // 计算所有帧速度变化量的方差
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enough!");
            // return false;
        }
    }
    // global sfm

    Quaterniond Q[frame_count + 1];           // 准备frame_count+1个位置来存放旋转Q
    Vector3d T[frame_count + 1];              // 准备frame_count+1个位置来存放平移T
    map<int, Vector3d> sfm_tracked_points;    // 准备个map存放[feature_id，特征点]
    vector<SFMFeature> sfm_f;                 // 准备个vector存放sfm出来的特征
    for (auto &it_per_id : f_manager.feature) // 遍历滑窗内所有特征
    {
        int imu_j = it_per_id.start_frame - 1; // 开一个计数器
        SFMFeature tmp_feature;                // 建立一个临时sfmfeature
        tmp_feature.state = false;             // sfmfeature状态先置为否，这个状态代表是否三角化
        tmp_feature.id = it_per_id.feature_id; // sfmfeature的id置为特征id
        // 遍历该特征的历史观测（归一化平面的点坐标），将存到tmp_feature当中
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            // SFMFeature的observation中，存储了一个空间点应的所有的观测帧id和归一化平面点
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // 在滑窗内找一帧质量好的作为参考帧，计算其与滑窗内最后一帧的相对位姿
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    // 创建一个sfm器，利用刚才找到的第l帧和滑窗最后一帧的相对位姿进行全局sfm
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        // 全局sfm失败了，将最老的帧丢弃
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    // 遍历每个图像帧
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i]) // 窗口内的帧位姿态都已经sfm出来了，直接赋值，设为关键帧
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); // 如果不是窗口内的帧，而是时间戳大于窗口内第i帧时间戳，以窗口内第i帧位姿为初值
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points) // 遍历一下图像帧的特征点，找到sfm出来的对应空间点，为PnP做准备
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 求解PnP，这里可以看出，有一帧要是没求解出来，整个initialStructure就算失败了
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // 在获取了sfm初始化结果以后，就要做惯性相关的初始化了
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    // 惯性初始化，求解bias和重力方向，完成视觉-惯性对齐
    // 送进这个函数是estimator滑窗中的陀螺零偏、重力引用，以及用于存放尺度因子的x
    // 这里面都是构造残差项，通过令error=0构造Hx=b线性方程，利用乔里斯基分解直接求解线性方程，不是通过非线性优化求解的
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 更新状态
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }
    double s = (x.tail<1>())(0);
    // 根据新的bias重新积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 根据新的尺度因子重新放缩平移
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // 更新速度
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/// @brief 找滑窗内匹配足够多的一个历史帧，并计算其与最新一帧的相对位姿
/// @param relative_R
/// @param relative_T
/// @param l
/// @return
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // 获取窗口里第i帧和最后一帧之间的特征关联
        if (corres.size() > 20)                              // 要是大于20个匹配，就接着进行视差大小的判断
        {
            double sum_parallax = 0;
            double average_parallax;
            // 算一下特征的平均视差大小
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 平均视差大于一定数值，使用motion_estimator求解第i帧和第WINDOW_SIZE帧(就是滑窗最后一帧)之间的位姿变换
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // 计算成功了！
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        // ROS_INFO(" big translation");
        // return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        // ROS_INFO(" big z translation");
        // return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

/// @brief 优化状态量
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    // ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if (!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            // ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if (STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    // printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    // printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    // printf("frame_count: %d \n", frame_count);

    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    // printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    // printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}
/// @brief 预测当前帧特征点在下一帧的位置
void Estimator::predictPtsInNextFrame()
{
    // printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);                   // 将CurT赋予当前位姿，即第[frame_count]个位姿
    getPoseInWorldFrame(frame_count - 1, prevT); // 将prevT赋予上一帧位姿，即第frame_count-1个位姿
    nextT = curT * (prevT.inverse() * curT);     // 利用上两帧之间的相对位姿预测下一帧的位姿
    map<int, Eigen::Vector3d> predictPts;        // 存储预测的特征点

    for (auto &it_per_id : f_manager.feature) // 遍历特征管理中的每个特征点
    {
        if (it_per_id.estimated_depth > 0) // 特征点深度大于0才能用
        {
            int firstIndex = it_per_id.start_frame;                                         // 发现该特征点的第一帧的索引
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1; // 发现该特征点的最后一帧的索引
            // printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) // 只有当该特征被观测到超过两次，且最后一次就是在当前帧被观测到
            {
                double depth = it_per_id.estimated_depth;                                                     // 估计的深度是在看到的第一帧相机系下的
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];            // 将相机系下的路标点转换到载体系下（第一帧时的）
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];                                     // 利用看到这个点的第一帧时的载体系位姿，将该点从载体系转换至世界系下
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3)); // 利用预测的下一帧位姿，将路标点转换至预测的下一帧载体系下
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);                                 // 将下一帧载体系下的路标点转换到相机系下
                int ptsIndex = it_per_id.feature_id;                                                          // 获取特征点的ID
                predictPts[ptsIndex] = pts_cam;                                                               // 将预测的路标点存入predicPts，ID-相机系下点坐标
            }
        }
    }
    featureTracker.setPrediction(predictPts); // 将预测特征点传给featureTracker
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    // return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
