#include <opencv2/core/types.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "OpenCVCalibration.h"

OpenCVCalibration::OpenCVCalibration(std::string calib_input)
{
    loadCalibrationParams(calib_input);
}

OpenCVCalibration::OpenCVCalibration(cv::Size chessboard_size, float chessboard_tile_size, int image_side, int calibration_flags, int view_undistorted)
    : chessboard_size_(chessboard_size), chessboard_tile_size_(chessboard_tile_size), image_side(image_side), calibration_flags(calibration_flags), view_undistorted(view_undistorted)
{
    assert(image_side == -1 || image_side == 0 || image_side == 1);
}

void OpenCVCalibration::splitImage(const cv::Mat &original_img, cv::Mat &split_img)
{

    static cv::Rect left_rect(0, 0, width_ , height_);
    static cv::Rect right_rect(width_ , 0, width_ , height_);
    if (image_side == 0)
    {
        split_img = original_img(left_rect);
    }
    else if (image_side == 1)
    {
        split_img = original_img(right_rect);
    }
    else
    {
        std::cerr << "Invalid corner flag in getSplitImage\n";
    }
}

void OpenCVCalibration::loadImages(std::vector<std::string> image_list)
{
    int num_images = static_cast<int>(image_list.size());
    cv::Size image_size; // make sure the images are the same size

    for (int i = 0; i < num_images; i++)
    {
        cv::Mat img = cv::imread(image_list[i], 0);
        if (img.empty())
        {
            std::cout << "Could not read image " << image_list[i] << ". Skipping the image\n";
            continue;
        }
        if (image_size == cv::Size())
        {
            image_size = img.size();
        }
        else if (image_size == img.size())
        {}
        else
        {
            std::cout << "The image " << image_list[i] << "has a size different from the other images. Skipping the image\n";
            continue;
        }
        calibration_images_.push_back(img.clone());
    }
    std::cout << "Successfully read " << calibration_images_.size() << " images.\n";
}

void OpenCVCalibration::runCalibration()
{
    findChessboardCornersFromImages();

    std::vector<std::vector<cv::Point3f>> object_points; // world-space location of the corners

    CalibUtils::calculateKnownChessboardPositions(static_cast<int>(calibration_images_.size()), chessboard_size_, chessboard_tile_size_, object_points);

    std::vector<cv::Mat> r_vect, t_vect;
    std::cout << "Running calibration on saved images\n";
    rep_error = cv::calibrateCamera(object_points, calibration_points_, cv::Size(width_, height_), camera_matrix, dist_coeffs, r_vect, t_vect, std_dev_intrinsics, std_dev_extrinsics, per_view_errors, calibration_flags, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
    
    if (!(cv::checkRange(camera_matrix) && cv::checkRange(dist_coeffs))) std::cout << "Warning: camera matrix or distortion coefficients has NaN or infinite elements\n";
    
    calibrated = true;
    std::cout << "Calibrated! Reprojection error: " << rep_error << std::endl;
    initialiseUndistort(1.0);
    if (view_undistorted) compareUndistortion();
}

void OpenCVCalibration::findChessboardCornersFromImages()
{
    std::vector<int> not_found_indices;
    std::cout << "Finding more accurate corners on saved calibration images...\n";
    for (int i = 0; i < static_cast<int>(calibration_images_.size()); i++)
    {
        std::vector<cv::Point2f> found_points;
        bool found = cv::findChessboardCornersSB(calibration_images_[i], chessboard_size_, found_points, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_ACCURACY | cv::CALIB_CB_MARKER);
        if (found)
        {
            // refine found points to subpixel accuracy -- use if using findChessboardCorners/other approx methods. findChessboardCornersSB returns more accurate subpixel locations directly
            // cv::cornerSubPix(frame, found_points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.0001)); // using same settings as https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
            calibration_points_.push_back(found_points);
        }
        else
        {
            not_found_indices.push_back(i);
            std::cout << "Chessboard could not be found on saved calibration image " << i << "!\n";
        }
        
    }
    // erase images if chessboard not found
    for (int j = static_cast<int>(not_found_indices.size()) - 1; j >= 0; j--)
    {
        calibration_images_.erase(calibration_images_.begin() + j);
    }
}

void OpenCVCalibration::compareUndistortion()
{
    // show sample undistorted images
    std::vector<cv::Mat> combined(calibration_images_.size());

    int nview = (view_undistorted == -1 || view_undistorted >= static_cast<int>(calibration_images_.size())) ? static_cast<int>(calibration_images_.size()) : view_undistorted;

    for (int i = 0; i < nview; i++) // static_cast<int>(calibration_images_.size())
    {
        cv::Mat undistorted;
        undistort(calibration_images_[i], undistorted);
        cv::hconcat(calibration_images_[i], undistorted.clone(), combined[i]);

        std::ostringstream oss;
        oss << std::setprecision(3) << "Comparison of image " << i << " - Reprojection error: " << per_view_errors.at<double>(i, 0);
        cv::imshow(oss.str(), combined[i]);
    }
    if (nview > 0)
    {
        std::cout << "Displaying sample undistorted image (Undistorted versions on the right), press any key to continue\n";
        cv::waitKey(0);
    }
}

void OpenCVCalibration::startCalibrationLoop(cv::VideoCapture vid_cap, bool save_images)
{
    std::cout << "\nCalibration started! With the image window in the foreground: \nSpacebar - saves the current image to use as a calibration image (calibration board must be detected). \nEnter - runs the calibration on the saved images (minimum 10 images). At least 20 images are recommended.\nq or Esc - exit the program.\n";
    cv::Mat original_img, rgbframe, frame, annotated_frame;
    vid_cap.read(rgbframe);
    width_ = (image_side == 1 || image_side == 0) ? rgbframe.cols / 2 : rgbframe.cols;
    height_ = rgbframe.rows;
    while (true)
    {
        if (!vid_cap.read(original_img))
        {
            std::cerr << "Could not read from VideoCapture device \n";
            break;
        }
    
        if (image_side == 0 || image_side == 1)
        {
            splitImage(original_img, rgbframe);
        }
        else if (image_side == -1)
        {
            rgbframe = original_img;
        }
        // work with monochrome to be faster?
        // cv::extractChannel(rgbframe, frame, 0);
        frame = rgbframe;

        std::vector<cv::Point2f> found_points;
        bool found = cv::findChessboardCornersSB(frame, chessboard_size_, found_points, cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_MARKER);
        frame.copyTo(annotated_frame);
        cv::drawChessboardCorners(annotated_frame, chessboard_size_, found_points, found);
        cv::imshow("Camera Feed [Calibration]", annotated_frame);

        char keyboard_char_pressed = static_cast<char>(cv::waitKey(1000 / 50));

        switch (keyboard_char_pressed)
        {
            case ' ': // space key -> save current frame
                if (found)
                {
                    calibration_images_.push_back(frame.clone());
                    std::cout << "Found chessboard; Calibration Images Count: " << calibration_images_.size() << std::endl;
                    if (save_images)
                    {
                        std::ostringstream oss;
                        if (image_side == 0) oss << "left-";
                        else if (image_side == 1) oss << "right-";
                        else oss << "image-";
                        oss << calibration_images_.size() << ".png";
                        cv::imwrite(oss.str(), frame);
                    }
                }
                else
                {
                    std::cout << "Chessboard was not in image. Try again\n";
                }
                
                break;
            case 13: // enter key -> run calibration
                if (calibration_images_.size() > 10)
                {   
                    runCalibration();
                    return;
                }
                else
                {
                    std::cout << "At least 10 images required for calibration! Take more calibration images first using spacebar to save the current image\n";
                }
                break;
            case 27: case 'q': // escape or q -> exit 
                return;
            default:
                break;
        }
    }
}

void OpenCVCalibration::saveCalibrationParams(std::string fname)
{
    std::cout << "Saving Calibration Parameters to " << fname << std::endl;
    std::cout << "Distortion Coefficients: " << dist_coeffs  << std::endl;
    std::cout << "Camera Matrix: " << camera_matrix << std::endl;

    cv::FileStorage fs(fname, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        std::ostringstream oss;
        std::string calibration_flags_string;
        CalibUtils::getCalibrationFlagsString(calibration_flags, calibration_flags_string);
        oss << "Calibration file generated on " << CalibUtils::GetDateTime() << " with calibration flags " << calibration_flags_string;
        fs.writeComment(oss.str());
        fs << "reprojection_error" << rep_error;
        fs << "image_side" << image_side;
        fs << "image_width" << width_;
        fs << "image_height" << height_;
        fs << "camera_matrix" << camera_matrix;
        fs << "distortion_coefficients" << dist_coeffs;
        fs << "calibration_flags" << calibration_flags;
        fs.release();
        std::cout << "Calibration parameters successfully saved to " << fname << std::endl;
    }
    else
    {
        std::cerr << "Could not open output file " << fname << " to write calibration parameters to!\n";
    }
}

bool OpenCVCalibration::loadCalibrationParams(std::string fname)
{
    std::cout << "Loading calibration file " << fname << std::endl;
    cv::FileStorage fs(fname, cv::FileStorage::READ);
    std::string line;

    if (fs.isOpened())
    {
        fs["image_side"] >> image_side;
        fs["image_width"] >> width_;
        fs["image_height"] >> height_;
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs["calibration_flags"] >> calibration_flags;
        
        calibrated = true;
        fs.release();
        std::cout << "Loaded calibration file " << fname <<std::endl;
        return true;
    }
    else
    {
        std::cout << "Calibration params file " << fname << " could not be opened.\n";
        return false;
    }
    
}

bool OpenCVCalibration::isCalibrated()
{
    return calibrated;
}

void OpenCVCalibration::initialiseUndistort(double alpha)
{
    assert (alpha >= 0.0 && alpha <= 1.0);
    if(calibrated)
    {
        cv::Size image_size(width_, height_);
        cv::Mat R; // this is an optional rectification transformation in the object space. will be needed when stereo
        // use getOptimalNewCameraMatrix https://answers.opencv.org/question/101398/what-does-the-getoptimalnewcameramatrix-function-does/
        // then use initUndistortRectifyMap -> remap instead for faster processing!
        optimal_new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, alpha, image_size);
        cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, R, optimal_new_camera_matrix, image_size, CV_16SC2, undistort_map1, undistort_map2); // could try speed comparison with CV_32FC1
    }
    else
    {
        std::cerr << "Not calibrated yet! Perhaps try loading calibration parameters?\n";
    }
}

void OpenCVCalibration::undistort(const cv::Mat &input_img, cv::Mat &output_img)
{
    cv::remap(input_img, output_img, undistort_map1, undistort_map2, cv::INTER_LINEAR);
}