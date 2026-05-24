/**
 * @file transform_dpl.cpp
 * @brief Image and keypoint transformation utilities for LightGlue
 * @note Copied from supervins_estimator for loop_fusion module use
 */

#include "transform_dpl.h"

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w)
{
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    for (const cv::Point2f &kpt : kpts)
    {
        cv::Point2f normalizedKpt = (kpt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
}
