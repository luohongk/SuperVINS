/**
 * @file matcher_dpl.h
 * @brief LightGlue feature matcher for loop closure verification
 * @note Adapted from supervins_estimator/src/featureTracker/extractor_matcher_dpl.h
 *       Only Matcher_DPL class is included (Extractor not needed in loop module)
 */

#pragma once

#ifndef LOOP_MATCHER_DPL_H
#define LOOP_MATCHER_DPL_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include "transform_dpl.h"

// Extractor type enum (consistent with supervins_estimator/parameters.h)
enum LoopExtractorType
{
    LOOP_SUPERPOINT = 0,
    LOOP_DISK = 1
};

class Matcher_DPL
{
public:
    Matcher_DPL(unsigned int _num_threads = 1);
    void initialize(std::string matcherPath, int extractor_type_, float matchThresh_);

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

    float matchThresh = 0.5;
    float last_avg_match_score = 0.0f;
    std::vector<cv::Point2f> pre_process(std::vector<cv::Point2f> kpts, int h, int w);
    std::vector<std::pair<int, int>> post_process();
    std::vector<std::pair<int, int>> match_featurepoints(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1);

    int extractor_type = 0;
};

#endif // LOOP_MATCHER_DPL_H
