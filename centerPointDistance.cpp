#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

float centerPointDistance(Rect rect1, Rect rect2)
{
	float xCenter1, yCenter1, xCenter2, yCenter2;
	xCenter1 = rect1.x + float(rect1.width) / 2;
	yCenter1 = rect1.y + float(rect1.height) / 2;
	xCenter2 = rect2.x + float(rect2.width) / 2;
	yCenter2 = rect2.y + float(rect2.height) / 2;
	return sqrt(pow((xCenter2 - xCenter1), 2) + pow((yCenter2 - yCenter1), 2));
}