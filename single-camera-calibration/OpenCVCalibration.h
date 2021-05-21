#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "../CalibUtils.h"

/* 
references
https://github.com/gdp-drone/camera-calibration
https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
*/

class OpenCVCalibration
{
public:
    // empty constructor
    OpenCVCalibration() = default;
    // use this constructor if already calibrated, to load the calib input file into the member variables
    OpenCVCalibration(std::string calib_input);
    // use this constructor if calibrating
    OpenCVCalibration(cv::Size chessboard_size, float chessboard_tile_size, int image_side, int calibration_flags, int view_undistorted);

    void saveCalibrationParams(std::string fname);
    bool loadCalibrationParams(std::string fname);
    void splitImage(const cv::Mat &original_img, cv::Mat &split_img);
    void loadImages(std::vector<std::string> image_list);
    
    // main loop.    image_side = 0 > use left half of the img; = 1 > use right half; = -1 > use entire img. save_images -> save images taken when spacebar pressed
    void startCalibrationLoop(cv::VideoCapture vid_cap, bool save_images);
    // calibration when enough images are taken
    void runCalibration();
    // runs a more accurate chessboard corner finder on calibration_images_ and places the results into calibration_points_;
    void findChessboardCornersFromImages();

    bool isCalibrated();
    // run after calibration to look at undistorted versions of all the images in calibration_images_;
    void compareUndistortion();
    // get the new camera matrix and map for undistorting
    void initialiseUndistort(double alpha); // alpha should be from 0 to 1, with 0 => all black parts from the undistorted image removed, 1 => no change to image (see https://answers.opencv.org/question/101398/what-does-the-getoptimalnewcameramatrix-function-does/)
    // undistort assumes you have already ran initialiseUndistort, no checks
    void undistort(const cv::Mat& input_img, cv::Mat& output_img);

public:
    cv::Size chessboard_size_;
    float chessboard_tile_size_;

    int image_side = -1; // 0 = left camera, 1 = right camera
    int width_ = 640; // of single camera image
    int height_ = 400;
    
    cv::Mat dist_coeffs = cv::Mat::zeros(1, 14, CV_64F);
    cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat optimal_new_camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
private:
    std::vector<cv::Mat> calibration_images_;
    std::vector<std::vector<cv::Point2f>> calibration_points_;
    bool calibrated = false;
    double rep_error = -1.; // reprojection error, set during calibration

    cv::Mat undistort_map1, undistort_map2;

    // calibration output
    cv::Mat per_view_errors, std_dev_intrinsics, std_dev_extrinsics;

    int calibration_flags;
    int view_undistorted;
};