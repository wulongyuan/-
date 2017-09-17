#include "BodyType.h"
#include "BodyDetect.h"
#include "ImageSegmentation.h"



bool sortCountersArea(vector<cv::Point> A, vector<cv::Point> B)
{
	return (contourArea(A) > contourArea(B));
}

bool sortX(cv::Point2f A, cv::Point2f B)
{
	return (A.x < B.x);
}

std::vector<cv::Point2f> skeletonBranchPoints(const cv::Mat &thinSrc, unsigned int raudis, unsigned int thresholdMax, unsigned int thresholdMin)
{
	assert(thinSrc.type() == CV_8UC1);
	cv::Mat dst;
	thinSrc.copyTo(dst);
	filterOver(dst);
	int width = dst.cols;
	int height = dst.rows;
	cv::Mat tmp;
	dst.copyTo(tmp);
	std::vector<cv::Point2f> branchpoints;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (*(tmp.data + tmp.step * i + j) == 0)
			{
				continue;
			}
			int count = 0;
			for (int k = i - raudis; k < i + raudis + 1; k++)
			{
				for (int l = j - raudis; l < j + raudis + 1; l++)
				{
					if (k < 0 || l < 0 || k>height - 1 || l>width - 1)
					{
						continue;

					}
					else if (*(tmp.data + tmp.step * k + l) == 1)
					{
						count++;
					}
				}
			}

			if (count > thresholdMax)
			{
				cv::Point2f point(j, i);
				branchpoints.push_back(point);
			}
		}
	}
	return branchpoints;
}



vector<cv::Point2f> skeletonEndPoints(cv::Mat &src)
{
	cv::Mat dst;

	vector<cv::Point2f> endpoints;
	cv::Mat k(3, 3, CV_8UC1);

	k.at<uchar>(0, 0) = 1;
	k.at<uchar>(1, 0) = 1;
	k.at<uchar>(2, 0) = 1;
	k.at<uchar>(0, 1) = 1;
	k.at<uchar>(1, 1) = 10;
	k.at<uchar>(2, 1) = 1;
	k.at<uchar>(0, 2) = 1;
	k.at<uchar>(1, 2) = 1;
	k.at<uchar>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_8UC1, k);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 1; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) == 11)
			{
				endpoints.push_back(cv::Point2f(j, i));
			}
		}
	}
	return endpoints;
}


vector<cv::Point2f> deleteTooNearPoints(vector<cv::Point2f> src)
{
	vector<cv::Point2f> dst;

	if (src.size() > 0)
	{
		dst.push_back(src[0]);

		for (int i = 1; i < src.size(); i++)
		{
			int count = 0;
			for (int j = 0; j < j - 1; j++)
			{
				if (sqrt(pow((src[i].x - src[j].x), 2) + pow((src[i].y - src[j].y), 2)) < 5)
				{
					count++;
				}
			}
			if (count < 1)
			{
				dst.push_back(src[i]);
			}
		}
	}
	return dst;
}


vector<cv::Point2f> calcBodyWide(cv::Mat &bodyThreshold, cv::Point2f Center)
{
	vector<cv::Point2f> widePoint;
	for (int i = (Center.x) - 1; i >= 0; i--)
	{
		if (bodyThreshold.at<uchar>(Center.y, i) == 0)
		{
			widePoint.push_back(cv::Point2f(i,Center.y));
			break;
		}
	}

	if (widePoint.size() < 1)
	{
		widePoint.push_back(cv::Point2f(0,Center.y));
	}

	for (int i = (Center.x) + 1; i < bodyThreshold.size().width; i++)
	{
		if (bodyThreshold.at<uchar>(Center.y, i) == 0)
		{
			widePoint.push_back(cv::Point2f(i,Center.y));
			break;
		}
	}

	if (widePoint.size() < 2)
	{
		widePoint.push_back(cv::Point2f(bodyThreshold.size().width - 1, Center.y));
	}
	return widePoint;
}





skeleton FromEdgePoints(vector<cv::Point2f> &skeletonEndPoints, vector<cv::Point2f> &skeletonBranchPoints, cv::Point2f Center, cv::Mat &bodyThreshold, vector<cv::Point> contours)
{

	skeleton skeletonData;
	skeletonData._heart = Center;

	vector<cv::Point2f> test1 = skeletonEndPoints;

	skeletonEndPoints = deleteTooNearPoints(skeletonEndPoints);


	vector<cv::Point2f> test2 = skeletonEndPoints;

	skeletonBranchPoints = deleteTooNearPoints(skeletonBranchPoints);



	cv::Point lowestPoint(0, 0);
	int distance = 0;
	//推算此人的最低点（包括手臂举起的长度）（可能推算身高没有实际意义）
	for (int i = 0; i < skeletonEndPoints.size(); i++)
	{
		if (skeletonEndPoints[i].y > distance)
		{
			lowestPoint = skeletonEndPoints[i];
			distance = lowestPoint.y;
		}
	}


	//判断交叉点的类型
	for (int i = 0; i < skeletonBranchPoints.size(); i++)
	{
		//可能是胸部节点
		if (skeletonBranchPoints[i].y < Center.y)
		{
			if (abs(skeletonData.bodyPoint[BodyData_chest].x - Center.x) > abs(skeletonBranchPoints[i].x - Center.x))	//判断最有可能的胸部节点
			{
				skeletonData.bodyPoint[BodyData_chest] = skeletonBranchPoints[i];
			}
		}
		//可能是腹部节点
		else
		{
			//若腹部节点太靠下则抛弃
			if (skeletonBranchPoints[i].y > Center.y + abs(Center.y - lowestPoint.y) * 3.0 / 5.0)
			{
				skeletonData.bodyPoint[BodyData_hip] = cv::Point(0, 0);
				continue;
			}
			if (sqrt(pow(skeletonData.bodyPoint[BodyData_hip].x - Center.x,2)+ pow(skeletonData.bodyPoint[BodyData_hip].y - (Center.y*1.3), 2)) > sqrt(pow(skeletonBranchPoints[i].x - Center.x,2)+ pow(skeletonBranchPoints[i].y - (Center.y*1.3), 2)))	//判断最有可能的腹部节点
			{
				skeletonData.bodyPoint[BodyData_hip] = skeletonBranchPoints[i];
			}
		}
	}

	vector<cv::Point2f> bodyWide;
	bodyWide = calcBodyWide(bodyThreshold, Center);



	//判断头部位置
	if (skeletonData.bodyPoint[BodyData_chest] != cv::Point2f(0, 0))
	{
		//以胸口作为判断标准
		for (int i = 0; i < skeletonEndPoints.size(); i++)
		{
			if (skeletonEndPoints[i].y > skeletonData.bodyPoint[BodyData_chest].y)
			{
				//在身体宽度范围之外，抛弃
				continue;
			}
			else
			{
				//找出离身体最近的点
				if (abs(skeletonEndPoints[i].x - skeletonData.bodyPoint[BodyData_chest].x) < abs(skeletonData.bodyPoint[BodyData_head].x - skeletonData.bodyPoint[BodyData_chest].x))
				{
					if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
					{
						if (sqrt(pow(skeletonEndPoints[i].x - skeletonData.bodyPoint[BodyData_chest].x,2)+ pow(skeletonEndPoints[i].y - skeletonData.bodyPoint[BodyData_chest].y, 2)) < sqrt(pow(skeletonData.bodyPoint[BodyData_head].x - skeletonData.bodyPoint[BodyData_chest].x, 2) + pow(skeletonData.bodyPoint[BodyData_head].y - skeletonData.bodyPoint[BodyData_chest].y, 2)))
						{
							skeletonData.bodyPoint[BodyData_head] = skeletonEndPoints[i];
						}
					}
					else
					{
						skeletonData.bodyPoint[BodyData_head] = skeletonEndPoints[i];
					}

				}
			}
		}
	}
	//否则以中心点作为判断标准（可能不准）
	else
	{
		//以胸口作为判断标准
		for (int i = 0; i < skeletonEndPoints.size(); i++)
		{
			if (skeletonEndPoints[i].y > Center.y)
			{
				//在身体宽度范围之外，抛弃
				continue;
			}
			else
			{
				//找出离身体最近的点
				if (abs(skeletonEndPoints[i].x - Center.x) < abs(skeletonData.bodyPoint[BodyData_head].x - Center.x))
				{
					if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
					{
						if (abs(skeletonEndPoints[i].y - Center.y) < abs(skeletonData.bodyPoint[BodyData_head].y - Center.y))
						{
							skeletonData.bodyPoint[BodyData_head] = skeletonEndPoints[i];
						}
					}
					else
					{
						skeletonData.bodyPoint[BodyData_head] = skeletonEndPoints[i];
					}
				}
			}
		}
	}

	//去除已配对的点
	vector<cv::Point2f>::iterator it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_head]);
	if (it != skeletonEndPoints.end())
		skeletonEndPoints.erase(it);


	//去除可能错误的点
	if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
	{
		for (it = skeletonEndPoints.begin(); it != skeletonEndPoints.end();)
		{
			if (it->x > bodyWide[0].x && it->x < bodyWide[1].x && it->y < Center.y && it->y > skeletonData.bodyPoint[BodyData_head].y)
			{
				it = skeletonEndPoints.erase(it);
			}
			else
			{
				it++;
			}
		}
	}



	//判断手脚（可能有误判）
	sort(skeletonEndPoints.begin(), skeletonEndPoints.end(), sortX);

	for (int i = 0; i < skeletonEndPoints.size(); i++)
	{
		if (skeletonEndPoints[i].y >(lowestPoint.y + Center.y) / 2.0)
			continue;
		if (skeletonEndPoints[i].x < Center.x)
		{
			skeletonData.bodyPoint[BodyData_leftHand] = skeletonEndPoints[i];
			break;
		}
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_leftHand]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (int i = skeletonEndPoints.size() - 1; i >= 0; i--)
	{
		if (skeletonEndPoints[i].y > (lowestPoint.y + Center.y) / 2.0)
			continue;
		if (skeletonEndPoints[i].x > Center.x)
		{
			skeletonData.bodyPoint[BodyData_rightHand] = skeletonEndPoints[i];
			break;
		}
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_rightHand]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (int i = 0; i < skeletonEndPoints.size(); i++)
	{
		if (skeletonEndPoints[i].y < (bodyThreshold.size().height + Center.y) / 2.0)
			continue;
		skeletonData.bodyPoint[BodyData_leftFoot] = skeletonEndPoints[i];
		break;
	}


	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_leftFoot]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (int i = skeletonEndPoints.size() - 1; i >= 0; i--)
	{
		if (skeletonEndPoints[i].y < (bodyThreshold.size().height + Center.y) / 2.0)
			continue;
		skeletonData.bodyPoint[BodyData_rightFoot] = skeletonEndPoints[i];
		break;
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_rightFoot]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	//给出胸部推定值

	if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
	{
		skeletonData.bodyPoint[BodyData_chest] = cv::Point2f((Center.x + skeletonData.bodyPoint[BodyData_head].x) / 2, (Center.y + skeletonData.bodyPoint[BodyData_head].y) / 2);
	}

	return skeletonData;
}


