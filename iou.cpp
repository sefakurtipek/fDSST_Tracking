#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <windows.h>
using namespace std;
//using namespace cv;

float iou(cv::Rect rect1, cv::Rect rect2)
{
	float x1, y1, x2, y2, z1, t1, z2, t2, xx1, yy1, xx2, yy2, area1, area2, h, w, intersection;
	x1 = rect1.x; y1 = rect1.y; x2 = rect1.width + rect1.x; y2 = rect1.height + rect1.y;
	z1 = rect2.x; t1 = rect2.y; z2 = rect2.width + rect2.x; t2 = rect2.height + rect2.y;
	area1 = (x2 - x1) * (y2 - y1);
	area2 = (z2 - z1) * (t2 - t1);
	xx1 = max(x1, z1);
	yy1 = max(y1, t1);
	xx2 = min(x2, z2);
	yy2 = min(y2, t2);
	h = max(0, yy2 - yy1);
	w = max(0, xx2 - xx1);
	intersection = w * h;
	return intersection / (area1 + area2 - intersection);
}