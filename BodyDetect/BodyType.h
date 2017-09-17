#pragma once
#include "BodyType.h"
#include <cv.hpp>
#include <iostream>
#include<deque>

using namespace std;


enum Enum_BodyData
{
	BodyData_head,//
	BodyData_leftHand,
	BodyData_rightHand,
	BodyData_leftFoot,
	BodyData_rightFoot,
	BodyData_chest,//胸
	BodyData_hip,//胯
	BodyData_len
};

struct skeleton
{
	cv::Point2f bodyPoint[BodyData_len] = { cv::Point2f(0, 0) };
	cv::Point2f _heart = cv::Point2f(0, 0);
	bool operator == (const skeleton &i);
	vector<cv::Point> skeletonContours;
};

struct PersonData
{
	int index = -1;
	float m_fTimes;//识别的时间
	skeleton skeletonData;
	deque<skeleton> oldSskeletonData;
	bool operator == (const PersonData &i);
};



struct jcBlockData
{
public:
	cv::Point dir;//方向
	cv::Point pos;//位置

	void operator= (jcBlockData& a)
	{
		dir = a.dir;//方向
		pos = a.pos;//位置
	}
};


struct BodyData
{
	unsigned int _index;
	std::vector<jcBlockData*> _keyBodyDts[BodyData_len];//各关键点，未识别的设置为NULL
	float m_fTimes;//识别的时间

	cv::Point _heart;//重心
	std::vector<cv::Point> m_contours;//轮廓
};

struct TornadoData
{
	int _index;
	Enum_BodyData _type;

	cv::Point _pos;
};