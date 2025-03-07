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

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>

struct Configuration
{
    std::string lightgluePath;
    std::string extractorPath;

    std::string extractorType;
    bool isEndtoEnd = true;
    bool grayScale = false;

    unsigned int image_size = 512;
    float threshold = 0.0f;

    std::string device;
    bool viz = false;
};

#endif // CONFIGURATION_H