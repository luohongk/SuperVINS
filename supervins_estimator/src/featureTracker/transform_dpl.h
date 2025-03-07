/**
 * @file main.cpp
 * @author letterso
 * @brief modified form OroChippw/LightGlue-OnnxRunner
 * @version 0.5
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <iostream>
#include <functional>

#include <opencv2/opencv.hpp>

cv::Mat NormalizeImage(cv::Mat &Image);

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w);

cv::Mat ResizeImage(const cv::Mat &Image, int size, float &scale, const std::string &fn = "max",
                    const std::string &interp = "area");

cv::Mat RGB2Grayscale(cv::Mat &Image);

#endif // TRANSFORM_H