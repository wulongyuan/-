#pragma once
#include<windows.h>
#include<iostream>
#include<cv.hpp>
#include "BodyType.h"

using namespace std;

std::vector<cv::Point2f> skeletonBranchPoints(const cv::Mat &thinSrc, unsigned int raudis = 4, unsigned int thresholdMax = 6, unsigned int thresholdMin = 4);
vector<cv::Point2f> skeletonEndPoints(cv::Mat &src);
vector<cv::Point2f> calcBodyWide(cv::Mat &bodyThreshold, cv::Point2f Center);

skeleton FromEdgePoints(vector<cv::Point2f> &skeletonEndPoints, vector<cv::Point2f> &skeletonBranchPoints, cv::Point2f Center, cv::Mat &bodyThreshold, vector<cv::Point> contours);
bool sortCountersArea(vector<cv::Point> A, vector<cv::Point> B);