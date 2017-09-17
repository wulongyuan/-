#pragma once
#include "BodyType.h"
#include <cv.hpp>
#include <iostream>
#include<deque>

using namespace std;



class CJcCalBody
{
public:
	bool recognizeImage(cv::Mat& ima);

	void GetBodyData(std::vector<BodyData> &BodyArr);
	void GetTornadoData(std::vector<TornadoData> &TornadoArr);
private:
	vector<PersonData> PersonInformation;
	int indexNum = 0;
};