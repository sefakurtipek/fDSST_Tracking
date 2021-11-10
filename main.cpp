#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include "fdssttracker.hpp"
#include "iou.h"
#include "centerPointDistance.h"
#include <windows.h>

using namespace std;
using namespace cv;
std::vector <cv::Mat> imgVec;


bool mouse_is_pressing = false;
int start_x, start_y, end_x, end_y;
int step = 0;
Mat img_color;


void swap(int* v1, int* v2) {
	int temp = *v1;
	*v1 = *v2;
	*v2 = temp;
}


void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
	Mat img_result = img_color.clone();
	if (event == EVENT_LBUTTONDOWN) {
		step = 1;
		mouse_is_pressing = true;
		start_x = x;
		start_y = y;
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (mouse_is_pressing) {
			end_x = x;
			end_y = y;
			step = 2;
		}
	}
	else if (event == EVENT_LBUTTONUP) {
		mouse_is_pressing = false;
		end_x = x;
		end_y = y;
		step = 3;
	}
}

int main(int argc, char* argv[]) {
	if (argc > 5) return -1;
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	int count = 1;
	FDSSTTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB); // tracker initialization

	enum dataType { frameBased, videoBased, webcamBased };
	dataType choice = frameBased;  // data type and video path need to be chosen by user /////////////////////////////////////////////////////////////////////////
	string videoPath, videoPath2, imgPath;
	videoPath = "C:\\Users\\sefa.kurtipek\\source\\repos\\fDSST_Tracking\\videoplayback.mp4";
	videoPath2 = "C:\\Users\\sefa.kurtipek\\source\\repos\\fDSST_Tracking\\videoplayback.mp4";
	imgPath = "C:\\Users\\sefa.kurtipek\\source\\repos\\fDSST_Tracking\\dataset\\tc_Logo_ce\\"; // data image path is chosen
	if (choice == frameBased) {  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		Mat processImg, colorImg;
		char name[7];
		//get init target box params from information file
		ifstream initInfoFile;
		string fileName = imgPath + "groundtruth.txt";
		initInfoFile.open(fileName);
		string line;
		getline(initInfoFile, line);
		float initX, initY, initWidth, initHeight, newXPoint, newYPoint, newWidth, newHeight, iouResult = 0, iouResultSum = 0, duration = 0, centerDistance = 0, centerDistanceSum = 0;
		char ch;
		istringstream ss(line);
		ss >> initX, ss >> ch;

		ss >> initY, ss >> ch;
		ss >> initWidth, ss >> ch;
		ss >> initHeight, ss >> ch;
		cv::Rect initRect = cv::Rect(initX, initY, initWidth, initHeight);

		for (;;)
		{
			auto t_start = clock();
			sprintf_s(name, "%04d", count);
			std::string imgFinalPath = imgPath + std::string(name) + ".jpg";
			processImg = cv::imread(imgFinalPath, IMREAD_GRAYSCALE);
			colorImg = cv::imread(imgFinalPath, IMREAD_COLOR);

			if (processImg.empty()) { // error handling
				cout << "no image has been created..." << endl;
			}
			if (processImg.empty()) // if image empty
			{
				break;
			}
			cv::Rect showRect;
			if (count == 1)		// initial step
			{
				tracker.init(initRect, processImg);
				showRect = initRect;
			}
			else {				// update step
				showRect = tracker.Update(processImg);
				getline(initInfoFile, line);
				std::istringstream ss(line);
				ss >> newXPoint, ss >> ch;
				ss >> newYPoint, ss >> ch;
				ss >> newWidth, ss >> ch;
				ss >> newHeight, ss >> ch;
				Rect newRect = Rect(newXPoint, newYPoint, newWidth, newHeight);
				iouResult = iou(newRect, showRect); // intersection over union function
				centerDistance = centerPointDistance(newRect, showRect);
			}
			auto t_end = clock();
			iouResultSum = iouResultSum + iouResult;
			centerDistanceSum = centerDistance + centerDistance;

			duration += (double)(t_end - t_start) / CLOCKS_PER_SEC;

			cv::rectangle(colorImg, showRect, cv::Scalar(0, 255, 0));
			cv::imshow("windows", colorImg);
			cv::waitKey(1);
			count++;
		}
		std::cout << "FPS: " << count / duration << endl;
		std::cout << "Average intersection over union: " << iouResultSum / (count - 1) << endl;
		std::cout << "Average center point distance: " << centerDistanceSum / (count - 1) << endl;

		system("pause");
	}
	else if (choice == videoBased) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		// Create a VideoCapture object and open the input file
		VideoCapture capture(videoPath2);    // video captures from video
		int numberOfTotalFrames = capture.get(CAP_PROP_FRAME_COUNT);
		if (!capture.isOpened()) {			// Check if camera opened successfully
			cout << "Error opening video stream or file" << endl;
			return -1;
		}
		cv::Mat initialFrame, processImg;
		capture >> initialFrame;
		cv::Rect2d initRect = cv::selectROI(initialFrame);

		double duration = 0;
		for (int i = 0; i < numberOfTotalFrames - 1; i++)
		{
			auto t_start = clock();
			Mat currentFrame;
			capture >> currentFrame;
			cvtColor(currentFrame, processImg, COLOR_RGB2GRAY);

			Rect showRect;
			if (count == 1)
			{
				tracker.init(initRect, processImg);
				showRect = initRect;
			}
			else {
				showRect = tracker.Update(processImg);
			}
			auto t_end = clock();
			duration += (double)(t_end - t_start) / CLOCKS_PER_SEC;

			cv::rectangle(currentFrame, showRect, cv::Scalar(0, 255, 0));
			cv::imshow("Tracking Result", currentFrame);
			cv::waitKey(1);
			count++;
		}
		std::cout << "FPS: " << count / duration << "\n";

		system("pause");
	}
	else if (choice == webcamBased) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		VideoCapture cap(videoPath);		// video captures from video
		Mat initialFrame;
		if (!cap.isOpened()) {
			cout << "Video is not opened";
			return -1;
		}
		namedWindow("Color", 1);
		setMouseCallback("Color", mouse_callback);
		while (step <= 3)
		{
			cap >> initialFrame;
			//cap.read(img_color);
			if (initialFrame.empty()) {
				cout << "image empty";
				break;
			}
			switch (step)
			{
			case 1:
				circle(initialFrame, Point(start_x, start_y), 10, Scalar(0, 255, 0), -1);
				break;
			case 2:
				rectangle(initialFrame, Point(start_x, start_y), Point(end_x, end_y), Scalar(0, 255, 0), 2);
				break;
			case 3:
				if (start_x > end_x) {
					swap(&start_x, &end_x);
					swap(&start_y, &end_y);
				}
				step++;
				break;
			}
			imshow("Color", initialFrame);

			if (waitKey(25) >= 0)
				break;
		}
		destroyAllWindows();
		cv::Rect2d initRect = cv::Rect(start_x, start_y, end_x - start_x, end_y - start_y);

		double duration = 0;
		for (;;)
		{
			Mat currentFrame, processImg;
			cap >> currentFrame;
			cvtColor(currentFrame, processImg, COLOR_RGB2GRAY);
			Rect showRect;
			if (count == 1)
			{
				tracker.init(initRect, processImg);
				showRect = initRect;
			}
			else {
				showRect = tracker.Update(processImg);
			}
			cv::rectangle(currentFrame, showRect, cv::Scalar(0, 255, 0));
			cv::imshow("windows", currentFrame);
			cv::waitKey(1);
			count++;
		}
	}
	destroyAllWindows();
	return 0;

}