#pragma once
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>

namespace CalibUtils{
    std::string GetDateTime();
    bool readStringList(const std::string &filename, std::vector<std::string> &l);
    // calculates the corners/object_points to use in calibrateCamera / stereoCalibrate
    void calculateKnownChessboardPositions(int num_images, cv::Size chessboard_size, float chessboard_tile_size, std::vector<std::vector<cv::Point3f>> &corners);
    double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>> &object_points,
                                                    const std::vector<std::vector<cv::Point2f>> &image_points,
                                                    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    // gets the calibration flags as a human readable string
    void getCalibrationFlagsString(int calibration_flags, std::string& calibration_flags_string);
    /* detects the corners of the charuco board, similar to findChessboardCorners
    returns true if number of markers detected is > 4

    references:
    https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/calibrate_camera_charuco.cpp
    https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/tutorial_charuco_create_detect.cpp
    https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_board_charuco.cpp
    https://github.com/amy-tabb/calico/blob/master/src/camera_calibration.cpp
    */
    bool findCharucoBoardCorners(cv::Mat image, const cv::Ptr<cv::aruco::CharucoBoard> board, std::vector<cv::Point2f>& charucoCorners, std::vector<int>& charucoIds);
} // namespace CalibUtils