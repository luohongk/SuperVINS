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

#ifndef VIZ2D_H
#define VIZ2D_H

#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

void plotMatches(const cv::Mat &figure, const std::vector<cv::Point2f> &kpts0,
                 const std::vector<cv::Point2f> &kpts1, int x_offset = 0,
                 std::vector<double> ratios = {1.0f, 1.0f}, const cv::Scalar &color = cv::Scalar(0, 255, 0),
                 float lw = 1, int ps = 2)
{
    assert(kpts0.size() == kpts1.size());
    if (lw > 0)
    {
        for (unsigned int i = 0; i < kpts0.size(); i++)
        {
            // 因为画布是计算比率缩放的，所以点也要进行缩放，同时右侧的图x坐标要加上左边的图的宽
            cv::Point2f pt0 = kpts0[i] / ratios[0];
            cv::Point2f pt1 = kpts1[i] / ratios[1];
            pt1.x += x_offset;

            // 实现matplotlib.patches.ConnectionPatch中的clip_on效果
            if (pt0.x > x_offset || pt0.y > figure.rows ||
                pt1.x < x_offset || pt1.y > figure.rows)
            {
                continue;
            }

            cv::line(figure, pt0, pt1, color, lw, cv::LINE_AA);
            if (ps > 0)
            {
                cv::circle(figure, pt0, ps, color, -1);
                cv::circle(figure, pt1, ps, color, -1);
            }
        }
    }
}

cv::Mat plotImages(const std::vector<cv::Mat> &Images,
                   std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_pair,
                   const std::vector<std::string> &Titles = std::vector<std::string>(1, "image"),
                   int dpi = 100, bool adaptive = true, float pad = 0.01f, bool show = true)
{
    /*
    Func:
        Plot a set of images horizontally.
    Args:
        imgs: a list of cv::Mat images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        adaptive: whether the figure size should fit the image aspect ratios.
    */
    try
    {
        unsigned int n = static_cast<int>(Images.size());
        std::vector<double> ratios;
        std::vector<double> img_rescale;

        for (const auto &image : Images)
        {
            if (adaptive)
            {
                ratios.emplace_back(static_cast<double>(image.cols) / image.rows); // W / H
            }
            else
            {
                ratios.emplace_back(4.0 / 3.0);
            }
        }

        // 整个图像集的绘图尺寸。它的宽度是所有图像宽高比之和乘以4.5，高度固定为4.5
        double totalRatio = std::accumulate(ratios.begin(), ratios.end(), 0.0);
        double figureWidth = totalRatio * 4.5;
        cv::Size2f figureSize((static_cast<double>(figureWidth)) * dpi, 4.5 * dpi);
        cv::Mat figure(figureSize, CV_8UC3);

        auto kpts0 = kpts_pair.first;
        auto kpts1 = kpts_pair.second;
        // std::cout << "[RESULT INFO] kpts0 Size : " << kpts0.size() << std::endl;
        // std::cout << "[RESULT INFO] kpts1 Size : " << kpts1.size() << std::endl;

        int x_offset = 0;
        for (unsigned int i = 0; i < n; ++i)
        {
            const cv::Mat &image = Images[i];
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            const std::string &title = Titles[i];
            img_rescale.push_back(image.rows / figureSize.height);

            cv::Mat resized_image;
            cv::Rect roi(cv::Point(x_offset, 0), cv::Size(static_cast<int>(ratios[i] * figureSize.height),
                                                          figureSize.height));
            cv::resize(image, resized_image, roi.size());
            resized_image.copyTo(figure(roi));

            if (!title.empty())
            {
                cv::putText(figure, title, cv::Point(x_offset + 10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            }

            if (i == 0)
            {
                x_offset += resized_image.cols;
            }
        }

        plotMatches(figure, kpts0, kpts1, x_offset, img_rescale);

        if (show)
        {
            cv::imshow("Figure", figure);
            cv::waitKey(800);
            cv::destroyAllWindows();
        }

        return figure;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] PlotImagesError : " << e.what() << '\n';

        return cv::Mat();
    }
}

void plotKeypoints()
{
    /*
    Func:
        Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    */
}

#endif // VIZ2D_H