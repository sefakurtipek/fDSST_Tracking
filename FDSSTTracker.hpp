#pragma once
#include "tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
//using namespace cv;

class FDSSTTracker : public Tracker
{
public:
    // Constructor
    FDSSTTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // Initialize tracker
    virtual void init(const cv::Rect& roi, cv::Mat image);

    // Update position based on the new frame
    virtual cv::Rect Update(cv::Mat image);

    float interp_factor; // linear interpolation factor for adaptation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size

    int base_width; // initial ROI widt
    int base_height; // initial ROI height
    int scale_max_area; // max ROI size before compressing
    float scale_padding; // extra area surrounding the target for scaling
    float scale_step; // scale step for multi-scale estimation
    float scale_sigma_factor; // bandwidth of gaussian target

    int n_scales; // # of scaling windows
    int n_interp_scales; // of interpolation scales

    int num_compressed_dim;

    float scale_lr; // scale learning rate

    std::vector<float> scaleFactors; // all scale changing rate, from larger to smaller with 1 to be the middle
    std::vector<float> interp_scaleFactors;

    int scale_model_width; // the model width for scaling
    int scale_model_height; // the model height for scaling
    float currentScaleFactor; // scaling rate
    float min_scale_factor; // min scaling rate
    float max_scale_factor; // max scaling rate
    float scale_lambda; // regularization


protected:
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat x, float& peak_value);

    // train tracker with a single image
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat& image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

    // Compute the FFT Guassian Peak for scaling
    cv::Mat computeYsf();

    // Compute the hanning window for scaling
    cv::Mat createHanningMatsForScale();

    // Initialization for scales
    void dsstInit(const cv::Rect& roi, cv::Mat image);

    // Compute the F^l in the paper
    cv::Mat get_scale_sample(const cv::Mat& image);

    // Update the ROI size after training
    void update_roi();

    // Train method for scaling
    void train_scale(cv::Mat image, bool ini = false);

    cv::Mat resizeDFT(const cv::Mat& A, int real_scales);

    // Detect the new scaling rate
    cv::Point2i detect_scale(cv::Mat image);

    cv::Mat features_projection(const cv::Mat& src);
    cv::Mat _labCentroids;

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;

    cv::Mat _proj_tmpl;

    cv::Mat _num;
    cv::Mat _den;

    cv::Mat sf_den;
    cv::Mat sf_num;

    cv::Mat proj_matrix;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;

    cv::Mat s_hann;
    cv::Mat ysf;

};
