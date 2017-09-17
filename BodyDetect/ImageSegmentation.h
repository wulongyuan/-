#pragma once
#include <cv.hpp>
#include <iostream>

using namespace std;

void filterOver(cv::Mat thinSrc);
void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);
cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
cv::Mat cutGreenScreen(cv::Mat& src, int cutTop, int cutBottom);