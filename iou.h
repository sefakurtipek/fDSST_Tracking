#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <windows.h>
using namespace std;
using namespace cv;

float iou(Rect rect1, Rect rect2);