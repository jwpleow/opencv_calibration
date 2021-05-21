#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include "../single-camera-calibration/OpenCVCalibration.h"

/* 
references
https://github.com/gdp-drone/camera-calibration
https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
*/

class OpenCVStereoCalibration
{
public:
    // use this constructor to load calibration to undistort images
    OpenCVStereoCalibration(std::string stereo_calib_file, double alpha = -1);
    // use this constructor if performing stereo calibration. view_rectified: how many rectified images to view. -1 for all. alpha: see stereoRectify
    OpenCVStereoCalibration(cv::Size chessboard_size, float chessboard_tile_size, int calibration_flags, int view_rectified, double alpha = -1);
    void saveCalibrationParams(std::string fname);
    bool loadStereoCalibrationParams(std::string fname);
    void loadMonoCalibrationParams(std::string left_calib_params_fname, std::string right_calib_params_fname);

    // main loop.    image_side = 0 > use left half of the img; = 1 > use right half; = -1 > use entire img.
    void startCalibrationLoop(cv::VideoCapture vid_cap , bool save_images);
    // calibration when enough images are taken
    void runStereoCalibration();
    // load images into calibration_images_x
    void loadImages(std::vector<std::string> image_list);

    bool isCalibrated();

    // get the map for undistorting
    // alpha should be -1 (default scaling) or from 0 to 1, with 0 => all black parts from the undistorted image removed, 1 => all valid pixels retained (see https://answers.opencv.org/question/101398/what-does-the-getoptimalnewcameramatrix-function-does/)
    void initialiseStereoRectificationAndUndistortMap(double alpha);
    void calculateMatchedRoi();

    /* 
    undistort assumes you have already ran initialiseStereoRectificationAndUndistortMap! no checks. 
    side = 0 for left camera, 1 for right
    */
    void undistort(const cv::Mat &input_img, cv::Mat &output_img, int side);
    // draw ROI and horizontal lines on the image
    void annotateUndistortedImage(const cv::Mat& left, const cv::Mat& right, cv::Mat& annotated_undistorted_img);
    void cropRoi(const cv::Mat &input_left, const cv::Mat &input_right, cv::Mat &output_left, cv::Mat &output_right);

private:
    void splitImage(const cv::Mat &original_img, cv::Mat &left_img, cv::Mat &right_img);
    void combineImage(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat& combined_img);

    // runs a more accurate chessboard corner finder on calibration_images and places the results into calibration_points;
    void findChessboardCornersFromImages();

    // make sure initialise undistort is called first
    double computeEpipolarError();

    // run after calibration to look at undistorted and rectified versions of all the images in calibration_images_;
    void viewUndistortion();
public:
    int width_ = 640; // of single camera
    int height_ = 400;
    cv::Size chessboard_size_;
    float chessboard_tile_size_;
    

    bool calibrated = false;
    double rep_error = -1.0; // reprojection error, calculated during calibration
    double epipolar_error = -1.0;

    // valid regions of interest for each camera
    cv::Rect validRoi_1;
    cv::Rect validRoi_2;
    // roi's with matched height 
    cv::Rect validMatchedRoi_1;
    cv::Rect validMatchedRoi_2;

    // camera 1 intrinsics from single camera calibration
    std::unique_ptr<OpenCVCalibration> left_calib_params;

    // camera 2 intrinsics
    std::unique_ptr<OpenCVCalibration> right_calib_params;
    
    // stereo camera extrinsic parameters - see https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    cv::Mat R = cv::Mat::zeros(3, 3, CV_64F); // rotation matrix
    cv::Mat T = cv::Mat::zeros(3, 1, CV_64F); // translation vector
    cv::Mat E = cv::Mat::zeros(3, 3, CV_64F); // essential matrix - describes 2nd camera relative to 1st
    cv::Mat F = cv::Mat::zeros(3, 3, CV_64F); // fundamental matrix - same info as essential matrix, along with information about the intrinsics of both cameras to relate the two cameras in pixel coords (maps a point in one image to an epiline in the other)
    // rectify transforms https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    cv::Mat R1 = cv::Mat::zeros(3, 3, CV_64F); // R1 -> rotation matrix from 1st to 2nd camera https://py.plainenglish.io/the-depth-i-stereo-calibration-and-rectification-24da7b0fb1e0
    cv::Mat R2 = cv::Mat::zeros(3, 3, CV_64F); // R2 -> rotation matrix from 2nd to 1st camera
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F); // P1 -> projection/position matrix from 1st to 2nd camera
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F); // P2 -> position matrix from 2nd to 1st camera
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);  // disparity-to-depth mapping matrix - distance between cameras, focal length, etc to get disparity map

private:
    std::vector<cv::Mat> calibration_images_1;
    std::vector<cv::Mat> calibration_images_2;
    std::vector<std::vector<cv::Point2f>> calibration_points_1;
    std::vector<std::vector<cv::Point2f>> calibration_points_2;

    cv::Mat undistort_map_1_1, undistort_map_1_2; // camera 1 maps for undistorting
    cv::Mat undistort_map_2_1, undistort_map_2_2;

    cv::Mat per_view_errors; // error for each calibration image

    int calibration_flags;
    int view_rectified;

    float alpha_; // for rectification. see stereoRectify 
};