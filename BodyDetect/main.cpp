#define  _CRT_SECURE_NO_WARNINGS 
#include <iostream>
#include <vector>
#include <deque>
#include <windows.h>
#include "ImageSegmentation.h"
#include "BodyDetect.h"
#include "Body.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
using namespace cv;
using namespace std;



int main()
{
	CJcCalBody test;
	int cutTop = 30, cutBottom = 10;
	VideoCapture video;
	video.open("../abc3.mp4");

	Mat videoFrame, videoFrameResize;

	while (video.read(videoFrame))
	{
		resize(videoFrame, videoFrameResize, Size(480, 270));

		DWORD startTime = GetCurrentTime();

		Rect cutRect = Rect(0, cutTop, videoFrameResize.size().width, videoFrameResize.size().height - cutTop - cutBottom);

		Mat videoDisplay, videoDisplayResize;
		videoFrameResize(cutRect).copyTo(videoDisplay);

		Mat cutFrame = cutGreenScreen(videoFrameResize, cutTop, cutBottom);

		test.recognizeImage(cutFrame);

		std::vector<BodyData> BodyArr;
		test.GetBodyData(BodyArr);

		for (int i = 0; i < BodyArr.size(); i++)
		{

			for (int j = 0; j < 7; j++)
			{
				if (BodyArr[i]._keyBodyDts[j][0] != NULL)
					circle(videoDisplay, BodyArr[i]._keyBodyDts[j][0]->pos, 4, Scalar(0, 255, 0),-1);
			
			}

			stringstream s;
			s << BodyArr[i]._index;


			putText(videoDisplay, s.str(), Point(BodyArr[i]._heart.x + 5, BodyArr[i]._heart.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
			cout << BodyArr[i].m_fTimes << endl;
		}

		std::vector<TornadoData> TornadoArr;
		test.GetTornadoData(TornadoArr);

		for (int i = 0; i < TornadoArr.size(); i++)
		{
			circle(videoDisplay,TornadoArr[i]._pos, 4, Scalar(0, 0, 255), -1);
		}

		cv::imshow("test",videoDisplay);
		waitKey(1);

	}

}