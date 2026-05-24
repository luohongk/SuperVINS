/**
 * @file transform_dpl.h
 * @brief Image and keypoint transformation utilities for LightGlue
 * @note Copied from supervins_estimator for loop_fusion module use
 */

#pragma once

#ifndef LOOP_TRANSFORM_DPL_H
#define LOOP_TRANSFORM_DPL_H

#include <iostream>
#include <functional>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w);

#endif // LOOP_TRANSFORM_DPL_H
