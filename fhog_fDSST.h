#pragma once
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "sse.hpp"
#include <opencv2/core/core.hpp>

float* fhog_fDSST(float* I, int height, int width, int channel, int* h, int* w, int* d, int binSize = 4, int nOrients = 9, float clip = 0.2f, bool crop = false);
cv::Mat fhog_fDSST(const cv::Mat& input, int binSize = 4, int nOrients = 9, float clip = 0.2f, bool crop = false);
void change_format(float* des, float* source, int height, int width, int channel);
// wrapper functions if compiling from C/C++
inline void wrError(const char* errormsg) { throw errormsg; }
inline void* wrCalloc(size_t num, size_t size) { return calloc(num, size); }
inline void* wrMalloc(size_t size) { return malloc(size); }
inline void wrFree(void* ptr) { free(ptr); }