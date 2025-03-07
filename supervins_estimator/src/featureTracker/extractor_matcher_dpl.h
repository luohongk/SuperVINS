// #include <onnxruntime_cxx_api.h>
#include "../featureTracker/ort_include/onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <string>
#include "transform_dpl.h"
#include "../estimator/parameters.h"
#include <chrono>
#include <fstream>


class Extractor_DPL
{
    public:
    Extractor_DPL(unsigned int _num_threads=1);
    void initialize(std::string extractorPath,int extractor_type_);
    
    const unsigned int num_threads;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> Session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char *> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;
    std::vector<char *> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

    float scale = 1.0f;
    std::vector<std::vector<Ort::Value>> outputtensors;
    // std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
    // std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> descriptors_result;

    cv::Mat pre_process(const cv::Mat &Image, float &scale);
    std::pair<std::vector<cv::Point2f>, float *> post_process(std::vector<Ort::Value> tensor);
    std::pair<std::vector<cv::Point2f>, float *> extract_featurepoints(const cv::Mat &image);

    int extractor_type = 0;
    unsigned int IMAGE_SIZE_DPL=512; 
    

   
};

class Matcher_DPL
{
    public:
    Matcher_DPL(unsigned int _num_threads=1);
    void initialize(std::string matcherPath,int extractor_type_, float matchThresh_);

    const unsigned int num_threads;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> Session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char *> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;
    std::vector<char *> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

    float scale = 1.0f;
    std::vector<Ort::Value> outputtensors;
    // std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
    // std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> descriptors_result;


    float matchThresh=0.5;
    std::vector<cv::Point2f> pre_process(std::vector<cv::Point2f> kpts, int h, int w);
    std::vector<std::pair<int,int>> post_process();
    std::vector<std::pair<int,int>> match_featurepoints(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1);
    
    int extractor_type = 0;
};