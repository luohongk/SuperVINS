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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;
//数据缓存队列，先进先出
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
//缓冲锁
std::mutex m_buf;


/// @brief 左目图像回调函数
/// @param img_msg 
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    //上锁，防止这时候其他函数读取
    m_buf.lock();
    //将数据存入缓冲区
    img0_buf.push(img_msg);
    //解锁，这之后其他函数可以读取
    m_buf.unlock();
}

//右目图像回调函数，同左目类似
void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

//将ROS图像消息格式转化为OpenCV格式 cv::Mat
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
//Synchronize the left and right eye images and input the synchronized images into the estimator 
void sync_process()
{
    while(1)
    {
        //双目情况下
        //Under binocular conditions
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            //上锁，防止在同步过程中有新图像存入缓冲区
            //Lock to prevent new images from being stored in the buffer during synchronization
            m_buf.lock();

            //双目情况下左右目缓冲区都不为0才可以进行同步
            //In the case of binocular vision, synchronization can only be performed if the buffers of the left and right eyes are both equal to 0.
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                //缓冲区中最老的左右目图像时间戳
                 //The oldest left and right image timestamp in the buffer
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();

                // 0.003s sync tolerance
                // 最大允许0.003秒的同步延迟

                //最老左目图像比最老右目图像还要更早出现，更早的时间超过0.003s了，那这张最老左目图像就不要了，直接弹出
                //The oldest left eye image appears earlier than the oldest right eye image, and the earlier time is more than 0.003s. Then the oldest left eye image is no longer needed and will pop up directly.
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                //最老左目图像比最老右目图像还要更晚出现，更晚的时间超过0.003s了，那这张最老右目图像就不要了，直接弹出
                //The oldest left eye image appears later than the oldest right eye image, and the later time is more than 0.003s. Then the oldest right eye image is no longer needed and will pop up directly.
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                //最老的左右目图像相差不超过0.003，那就是处理它俩了，将这俩图像当作一帧，以左目图像时间戳、header为主，取出图像转化为cv::Mat
                //The difference between the oldest left and right eye images does not exceed 0.003, that is to process them. Treat these two images as one frame, mainly the left eye image timestamp and header, take out the image and convert it into cv::mat
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                }
            }
            //同步完了，可以解锁了
            //Synchronization is complete and can be unlocked
            m_buf.unlock();
            //图像不为空的话，输入到估计器里
            //If the image is not empty, enter it into the estimator
            if(!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        //单目情况
        //Monocular situation
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            //惯例，先上锁，防止读数据时有新数据插入
            //As a general rule, lock first to prevent new data from being inserted when reading data.
            m_buf.lock();
            //左目图像缓冲区不为空才能处理
            //The left eye image buffer can only be processed if it is not empty.
            if(!img0_buf.empty())
            {
                //直接取左目最老图像就好了
                //Just take the oldest left eye image directly
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            //解锁
            //Unlock
            m_buf.unlock();
            if(!image.empty())
                //将图像输入到估计器中，在输入图像的函数中，状态估计就已经开始了
                 //Input the image into the estimator. In the function of the input image, the state estimation has already started.
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

//IMU回调函数,imu_buf没用，直接把IMU数据往estimator里输入
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

//特征回调函数，在接收到新的特征时调用,这个函数在VINS-Mono中使用，在VINS-Fusion中feature_tracker内置在了estimator中
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    
    ros::init(argc, argv, "supervins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun supervins supervins_node [config file] \n"
               "for example: upervins supervins_node"
               "~/catkin_ws/src/SuperVINS/config/euroc/euroc_mono_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    //读取参数，包括相机参数、IMU参数、特征参数等等，读入到parameters.cpp这个文件里的变量里
    // parameters, including camera parameters, imu parameters, feature parameters, etc., into variables in the file parameters.cpp
    readParameters(config_file);

    // 路径配置
    // Path configuration
    std::string project_source_dir = PROJECT_SOURCE_DIR;
    extractor_weight_global_path = project_source_dir + "/" + extractor_weight_relative_path;
    matcher_weight_global_path = project_source_dir + "/" + matcher_weight_relative_path;
    VINS_RESULT_PATH = project_source_dir + "/" + VINS_RESULT_PATH;

    //给estimator设置参数，因为一些参数可能被优化，所以可能会重置参数，注意，如果开启了多线程模式，在setParameter()中就已经将状态估计函数放入一个独立线程运行了
    //Set parameters for the estimator. Because some parameters may be optimized, the parameters may be reset. Note that if the multi-thread mode is turned on, the state estimation function has been put into an independent thread to run in set parameter().
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);
    // 订阅IMU话题的接收器
    // Receiver that subscribes to imu topics
    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        // 接收来自IMU_TOPIC的IMU数据，每接收一个就送入estimator
        // Receive imu data from imu topic, and send each one to the estimator
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    //接收特征、图像
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    //接收系统重启、IMU切换、图像切换信号
    //Receive system restart, imu switching, image switching signals
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    //同步处理
    //Synchronization
    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;

    
    
}
