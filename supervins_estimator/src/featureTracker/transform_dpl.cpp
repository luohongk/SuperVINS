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

#include "transform_dpl.h"

cv::Mat NormalizeImage(cv::Mat &Image)
{
    cv::Mat normalizedImage = Image.clone();

    if (Image.channels() == 3)
    {
        cv::cvtColor(normalizedImage, normalizedImage, cv::COLOR_BGR2RGB);
        normalizedImage.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    }
    else if (Image.channels() == 1)
    {
        Image.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    }
    else
    {
        throw std::invalid_argument("[ERROR] Not an image");
    }

    return normalizedImage;
}

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w)
{
    cv::Size size(w, h);
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

cv::Mat ResizeImage(const cv::Mat &Image, int size, float &scale, const std::string &fn,
                    const std::string &interp)
{
    // Resize an image to a fixed size, or according to max or min edge.
    int h = Image.rows;
    int w = Image.cols;

    std::function<int(int, int)> func;
    if (fn == "max")
    {
        func = [](int a, int b)
        { return (std::max)(a, b); };
        ;
    }
    else if (fn == "min")
    {
        func = [](int a, int b)
        { return (std::min)(a, b); };
    }
    else
    {
        throw std::invalid_argument("[ERROR] Incorrect function: " + fn);
    }

    int h_new, w_new;
    if (size == 512 || size == 1024 || size == 2048)
    {
        scale = static_cast<float>(size) / static_cast<float>(func(h, w));
        h_new = static_cast<int>(round(h * scale));
        w_new = static_cast<int>(round(w * scale));
    }
    else
    {
        throw std::invalid_argument("Incorrect new size: " + std::to_string(size));
    }

    int mode;
    if (interp == "linear")
    {
        mode = cv::INTER_LINEAR;
    }
    else if (interp == "cubic")
    {
        mode = cv::INTER_CUBIC;
    }
    else if (interp == "nearest")
    {
        mode = cv::INTER_NEAREST;
    }
    else if (interp == "area")
    {
        mode = cv::INTER_AREA;
    }
    else
    {
        throw std::invalid_argument("[ERROR] Incorrect interpolation mode: " + interp);
    }

    cv::Mat resizeImage;
    cv::resize(Image, resizeImage, cv::Size(w_new, h_new), 0, 0, mode);

    return resizeImage;
}

cv::Mat RGB2Grayscale(cv::Mat &Image)
{
    cv::Mat resultImage;
    cv::cvtColor(Image, resultImage, cv::COLOR_RGB2GRAY);

    return resultImage;
}