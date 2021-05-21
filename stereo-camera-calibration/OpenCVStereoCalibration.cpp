#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include "OpenCVStereoCalibration.h"

OpenCVStereoCalibration::OpenCVStereoCalibration(cv::Size chessboard_size, float chessboard_tile_size, int calibration_flags, int view_rectified, double alpha)
    : chessboard_size_(chessboard_size), chessboard_tile_size_(chessboard_tile_size), calibration_flags(calibration_flags), view_rectified(view_rectified), alpha_(alpha)
{
}

OpenCVStereoCalibration::OpenCVStereoCalibration(std::string stereo_calib_file, double alpha)
: alpha_(alpha)
{
    left_calib_params = std::make_unique<OpenCVCalibration>();
    right_calib_params = std::make_unique<OpenCVCalibration>();
    loadStereoCalibrationParams(stereo_calib_file);
    initialiseStereoRectificationAndUndistortMap(alpha_);
}

void OpenCVStereoCalibration::splitImage(const cv::Mat &original_img, cv::Mat &left_img, cv::Mat &right_img)
{
    static cv::Rect left_rect(0, 0, width_ , height_);
    static cv::Rect right_rect(width_ , 0, width_ , height_);
    left_img = original_img(left_rect);
    right_img = original_img(right_rect);
}

void OpenCVStereoCalibration::combineImage(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &combined_img)
{
    cv::hconcat(left_img, right_img, combined_img);
}

void OpenCVStereoCalibration::runStereoCalibration()
{
    findChessboardCornersFromImages();

    std::vector<std::vector<cv::Point3f>> object_points; // world-space location of the corners

    CalibUtils::calculateKnownChessboardPositions(static_cast<int>(calibration_points_1.size()), chessboard_size_, chessboard_tile_size_, object_points);

    std::vector<cv::Mat> r_vect, t_vect;
    std::cout << "Running calibration on saved images\n";
    // test -- use this if initial calib sucked / not using an initial mono calib
    // left_calib_params->camera_matrix = cv::initCameraMatrix2D(object_points, calibration_points_1, cv::Size(width_, height_), 0);
    // right_calib_params->camera_matrix = cv::initCameraMatrix2D(object_points, calibration_points_2, cv::Size(width_, height_), 0);

    rep_error = cv::stereoCalibrate(object_points, calibration_points_1, calibration_points_2, left_calib_params->camera_matrix, left_calib_params->dist_coeffs,
                                    right_calib_params->camera_matrix, right_calib_params->dist_coeffs, cv::Size(width_, height_), R, T, E, F, per_view_errors, calibration_flags, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

    calibrated = true;
    std::cout << "Calibrated! Reprojection/RMS error: " << rep_error << std::endl;
    computeEpipolarError();
    initialiseStereoRectificationAndUndistortMap(alpha_);
    viewUndistortion();
}

void OpenCVStereoCalibration::findChessboardCornersFromImages()
{
    std::vector<int> not_found_indices;
    std::cout << "Finding more accurate corners on saved calibration images...\n";
    for (int i = 0; i < static_cast<int>(calibration_images_1.size()); i++)
    {
        std::vector<cv::Point2f> found_points_1;
        std::vector<cv::Point2f> found_points_2;
        bool found_1 = cv::findChessboardCornersSB(calibration_images_1[i], chessboard_size_, found_points_1, cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_ACCURACY | cv::CALIB_CB_MARKER | cv::CALIB_CB_EXHAUSTIVE);
        bool found_2 = cv::findChessboardCornersSB(calibration_images_2[i], chessboard_size_, found_points_2, cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_ACCURACY | cv::CALIB_CB_MARKER | cv::CALIB_CB_EXHAUSTIVE);
        if (found_1 && found_2)
        {
            // refine found points to subpixel accuracy -- if using findChessboardCorners/other approx methods. findChessboardCornersSB returns accurate subpixel locations directly
            // cv::cornerSubPix(frame, found_points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001)); // using same settings as https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
            calibration_points_1.push_back(found_points_1);
            calibration_points_2.push_back(found_points_2);
        }
        else
        {
            not_found_indices.push_back(i);
            std::cout << "Chessboard could not be found on saved calibration image pair " << i << "!\n";
        }
    }
    // erase images if chessboard not found
    for (int j = static_cast<int>(not_found_indices.size()) - 1; j >= 0; j--)
    {
        calibration_images_1.erase(calibration_images_1.begin() + j);
        calibration_images_2.erase(calibration_images_2.begin() + j);
    }
}

void OpenCVStereoCalibration::annotateUndistortedImage(const cv::Mat &left, const cv::Mat &right, cv::Mat &annotated_undistorted_img)
{
    std::vector<cv::Vec3f> lines[2];
    cv::Mat left_temp, right_temp;
    left.copyTo(left_temp);
    right.copyTo(right_temp);
    cv::cvtColor(left_temp, left_temp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_temp, right_temp, cv::COLOR_GRAY2BGR);
    // draw rectangles of the valid region of interests
    cv::rectangle(left_temp, validMatchedRoi_1, cv::Scalar(0, 0, 255), 3, 8);
    cv::rectangle(right_temp, validMatchedRoi_2, cv::Scalar(0, 0, 255), 3, 8);
    combineImage(left_temp, right_temp, annotated_undistorted_img);

    for (int j = 0; j < annotated_undistorted_img.rows; j += 16)
    {
        cv::line(annotated_undistorted_img, cv::Point(0, j), cv::Point(annotated_undistorted_img.cols, j), cv::Scalar(0, 255, 0), 1, 8);
    }
}

void OpenCVStereoCalibration::viewUndistortion()
{
    // show sample undistorted images from calibration images
    std::vector<cv::Mat> combined(calibration_images_1.size());
    std::vector<cv::Vec3f> lines[2];
    // no of images to view
    int nview = (view_rectified == -1 || view_rectified >= static_cast<int>(calibration_images_1.size())) ? static_cast<int>(calibration_images_1.size()) : view_rectified;

    for (int i = 0; i < nview; i++)
    {
        cv::Mat undistorted_1, undistorted_2;
        undistort(calibration_images_1[i], undistorted_1, 0);
        undistort(calibration_images_2[i], undistorted_2, 1);

        // draw rectangles of the valid region of interests
        cv::rectangle(undistorted_1, validRoi_1, cv::Scalar(0, 0, 255), 3, 8);
        cv::rectangle(undistorted_2, validRoi_2, cv::Scalar(0, 0, 255), 3, 8);
        combineImage(undistorted_1.clone(), undistorted_2.clone(), combined[i]);

        for (int j = 0; j < combined[i].rows; j += 16)
        {
            cv::line(combined[i], cv::Point(0, j), cv::Point(combined[i].cols, j), cv::Scalar(0, 255, 0), 1, 8);
        }
        std::ostringstream oss;
        oss << std::setprecision(3) << "Rectified image " << i << " - Reprojection errors - L: " << per_view_errors.at<double>(i, 0) << " R: " << per_view_errors.at<double>(i, 1);
        cv::imshow(oss.str(), combined[i]);
    }
    if (nview > 0)
    {
        std::cout << "Displaying undistorted and rectified versions of the saved calibration images, press any key to continue\n";
        cv::waitKey(0);
    }
}

void OpenCVStereoCalibration::startCalibrationLoop(cv::VideoCapture vid_cap, bool save_images)
{
    std::cout << "\nCalibration started! With the image window in the foreground: \nSpacebar - saves the current image to use as a calibration image (calibration board must be detected). \nEnter - runs the calibration on the saved images (minimum 10 images). At least 20 images are recommended.\nq or Esc - exit the program.\n";
    cv::Mat original_img, gray_img, left_img, right_img, annotated_left_img, annotated_right_img;
    vid_cap.read(original_img);
    width_ = original_img.cols / 2;
    height_ = original_img.rows;

    while (true)
    {
        if (!vid_cap.read(original_img))
        {
            std::cerr << "Could not read from VideoCapture device \n";
            break;
        }
        // work with monochrome to be faster?
        // cv::extractChannel(original_img, gray_img, 0);
        splitImage(original_img, left_img, right_img);

        std::vector<cv::Point2f> found_points_1;
        std::vector<cv::Point2f> found_points_2;
        bool found_1 = cv::findChessboardCorners(left_img, chessboard_size_, found_points_1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found_2 = cv::findChessboardCorners(right_img, chessboard_size_, found_points_2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        left_img.copyTo(annotated_left_img);
        right_img.copyTo(annotated_right_img);
        cv::drawChessboardCorners(annotated_left_img, chessboard_size_, found_points_1, found_1);
        cv::drawChessboardCorners(annotated_right_img, chessboard_size_, found_points_2, found_2);
        cv::Mat annotated_combined_image;
        combineImage(annotated_left_img, annotated_right_img, annotated_combined_image);
        cv::imshow("Camera Feed [Calibration]", annotated_combined_image);

        char keyboard_char_pressed = static_cast<char>(cv::waitKey(1000 / 50));

        switch (keyboard_char_pressed)
        {
            case ' ': // space key -> save current images
                if (found_1 && found_2)
                {
                    calibration_images_1.push_back(left_img.clone()); // does std::move work?
                    calibration_images_2.push_back(right_img.clone());

                    if (save_images) 
                    {
                        std::ostringstream oss1, oss2;
                        // std::string datetime = GetDateTime();
                        oss1 << "left-" << calibration_images_1.size() << ".png";
                        oss2 << "right-" << calibration_images_1.size() << ".png";
                        cv::imwrite(oss1.str(), left_img);
                        cv::imwrite(oss2.str(), right_img);
                    }
                    std::cout << "Found chessboard; Calibration Image Pairs Count: " << calibration_images_1.size() << std::endl;
                }
                else
                {
                    std::cout << "Chessboard was not found in both images. Try again\n";
                }

                break;
            case 13: // enter key -> run calibration
                if (calibration_images_1.size() > 10)
                {
                    runStereoCalibration();
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

void OpenCVStereoCalibration::saveCalibrationParams(std::string fname)
{
    std::cout << "Saving Calibration Parameters to " << fname << std::endl;

    cv::FileStorage fs(fname, cv::FileStorage::WRITE);

    if (fs.isOpened())
    {
         std::ostringstream oss;
	        
        std::string calibration_flags_string;
        CalibUtils::getCalibrationFlagsString(calibration_flags, calibration_flags_string);
        oss << "Calibration file generated on " << CalibUtils::GetDateTime() << " with calibration flags " << calibration_flags_string;
        fs.writeComment(oss.str());
        fs << "reprojection_error" << rep_error;
        fs << "epipolar_error" << epipolar_error;
        fs << "image_width" << width_;
        fs << "image_height" << height_;
        fs << "left_camera_matrix" << left_calib_params->camera_matrix;
        fs << "left_distortion_coefficients" <<  left_calib_params->dist_coeffs;
        fs << "right_camera_matrix" << right_calib_params->camera_matrix;
        fs << "right_distortion_coefficients" << right_calib_params->dist_coeffs;
        fs << "R" << R;
        fs << "T" << T;
        fs << "E" << E;
        fs << "F" << F;
        fs << "R1" << R1;
        fs << "R2" << R2;
        fs << "P1" << P1;
        fs << "P2" << P2;
        fs << "Q" << Q;
        fs.release();

        std::cout << "Calibration parameters successfully saved to " << fname << std::endl;
    }
    else
    {
        std::cerr << "Could not open output file " << fname << " to write calibration parameters to!\n";
    }
}

bool OpenCVStereoCalibration::loadStereoCalibrationParams(std::string fname)
{
    std::cout << "Loading stereo calibration parameter file " << fname << std::endl;
    cv::FileStorage fs(fname, cv::FileStorage::READ);

    if (fs.isOpened())
    {
        fs["image_width"] >> width_;
        fs["image_height"] >> height_;
        fs["left_camera_matrix"] >> left_calib_params->camera_matrix;
        fs["left_distortion_coefficients"] >> left_calib_params->dist_coeffs;
        fs["right_camera_matrix"] >> right_calib_params->camera_matrix;
        fs["right_distortion_coefficients"] >> right_calib_params->dist_coeffs;
        fs["R"] >> R;
        fs["T"] >> T;
        fs["E"] >> E;
        fs["F"] >> F;

        calibrated = true;
        fs.release();
        std::cout << "Loaded stereo calibration file " << fname << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

void OpenCVStereoCalibration::loadMonoCalibrationParams(std::string left_calib_params_fname, std::string right_calib_params_fname)
{
    left_calib_params = std::make_unique<OpenCVCalibration>(left_calib_params_fname);
    right_calib_params = std::make_unique<OpenCVCalibration>(right_calib_params_fname);
    assert(left_calib_params->isCalibrated() && right_calib_params->isCalibrated());
    assert(left_calib_params->image_side == 0 && right_calib_params->image_side == 1);
    std::cout << "left camera matrix: " << left_calib_params->camera_matrix << std::endl;
    std::cout << "right camera matrix: " << right_calib_params->camera_matrix << std::endl;
    std::cout << "left dist_coeffs: " << left_calib_params->dist_coeffs << std::endl;
    std::cout << "right dist_coeffs: " << right_calib_params->dist_coeffs << std::endl;
}

double OpenCVStereoCalibration::computeEpipolarError()
{ // taken from https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_calib.cpp
    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    std::vector<std::vector<cv::Vec3f>> lines(2);
    for (int i = 0; i < static_cast<int>(calibration_images_1.size()); i++)
    {
        int npt = static_cast<int>(calibration_points_1[i].size());
        cv::Mat imgpt[2];

        imgpt[0] = cv::Mat(calibration_points_1[i], true);
        // undistortPoints(imgpt[0], imgpt[0], left_calib_params->camera_matrix, left_calib_params->dist_coeffs, R1, P1);
        undistortPoints(imgpt[0], imgpt[0], left_calib_params->camera_matrix, left_calib_params->dist_coeffs, cv::Mat(), left_calib_params->camera_matrix);
        computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);

        imgpt[1] = cv::Mat(calibration_points_2[i], true);
        // undistortPoints(imgpt[1], imgpt[1], right_calib_params->camera_matrix, right_calib_params->dist_coeffs, R2, P2);
        undistortPoints(imgpt[1], imgpt[1], right_calib_params->camera_matrix, right_calib_params->dist_coeffs, cv::Mat(), right_calib_params->camera_matrix);
        computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
        for (int j = 0; j < npt; j++)
        { // calculate error using undistorted image points and the epilines
            double errij = std::abs(imgpt[0].at<cv::Point2f>(j, 0).x * lines[1][j][0] +
                                    imgpt[0].at<cv::Point2f>(j, 0).y * lines[1][j][1] + lines[1][j][2]) +
                           std::abs(imgpt[1].at<cv::Point2f>(j, 0).x * lines[0][j][0] +
                                    imgpt[1].at<cv::Point2f>(j, 0).y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    epipolar_error = err / npoints;
    std::cout << "Average Epipolar error: " << epipolar_error << std::endl;
    return epipolar_error;
}

void OpenCVStereoCalibration::loadImages(std::vector<std::string> image_list)
{   
    int num_images = static_cast<int>(image_list.size());
    cv::Size image_size; // make sure the images are the same size
    if (num_images % 2 != 0)
    {
        std::cout << "Error: the image list contains an odd number of images\n";
        return;
    }

    for (int i = 0; i < num_images / 2; i++)
    {
        cv::Mat left_img = cv::imread(image_list[i * 2], 0);
        cv::Mat right_img = cv::imread(image_list[i * 2 +1], 0);
        if (left_img.empty() || right_img.empty())
        {
            std::cout << "Could not read images " << image_list[i * 2] << " or " << image_list[i * 2 + 1] << ". Skipping the pair\n";
            continue;
        }
        if (image_size == cv::Size() && left_img.size() == right_img.size())
        {
            image_size = left_img.size();
        }
        else if (left_img.size() != image_size || left_img.size() != right_img.size())
        {
            std::cout << "The images " << image_list[i * 2] << " or " << image_list[i * 2 + 1] << " have size different from the first image size (or from each other). Skipping the pair\n";
            continue;
        }
        calibration_images_1.push_back(left_img.clone());
        calibration_images_2.push_back(right_img.clone());
    }
    std::cout << "Successfully read " << calibration_images_1.size() << " pairs of images.\n";
}
bool OpenCVStereoCalibration::isCalibrated()
{
    return calibrated;
}

void OpenCVStereoCalibration::initialiseStereoRectificationAndUndistortMap(double alpha)
{
    assert((alpha >= 0.0 && alpha <= 1.0) || alpha == -1.0);
    if (calibrated)
    {
        cv::Size image_size(width_, height_);
        cv::stereoRectify(left_calib_params->camera_matrix, left_calib_params->dist_coeffs, right_calib_params->camera_matrix, right_calib_params->dist_coeffs, image_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, alpha, image_size, &validRoi_1, &validRoi_2);

        // camera 1
        cv::initUndistortRectifyMap(left_calib_params->camera_matrix, left_calib_params->dist_coeffs, R1, P1, image_size, CV_16SC2, undistort_map_1_1, undistort_map_1_2); // could try speed comparison with CV_32FC1
        // camera 2
        cv::initUndistortRectifyMap(right_calib_params->camera_matrix, right_calib_params->dist_coeffs, R2, P2, image_size, CV_16SC2, undistort_map_2_1, undistort_map_2_2);
        calculateMatchedRoi();
    }
}

void OpenCVStereoCalibration::undistort(const cv::Mat &input_img, cv::Mat &output_img, int side)
{
    if (side == 0)
    {
        cv::remap(input_img, output_img, undistort_map_1_1, undistort_map_1_2, cv::INTER_LINEAR);
    }
    else if (side == 1)
    {
        cv::remap(input_img, output_img, undistort_map_2_1, undistort_map_2_2, cv::INTER_LINEAR);
    }
    else
    {
        std::cerr << "invalid side argument in undistort\n";
    }
}

void OpenCVStereoCalibration::cropRoi(const cv::Mat &input_left, const cv::Mat &input_right, cv::Mat &output_left, cv::Mat &output_right)
{
    assert(!validMatchedRoi_1.empty() && !validMatchedRoi_2.empty());
    output_left = input_left(validMatchedRoi_1);
    output_right = input_right(validMatchedRoi_2);
}

void OpenCVStereoCalibration::calculateMatchedRoi()
{
    // new valid regions of interest for each camera, ensuring epipolar lines are still intact, matching distances from centreline
    // match y loc
    int new_y_loc = std::max(validRoi_1.y, validRoi_2.y);
    // x loc for right img
    int new_x_loc_right = std::max(validRoi_2.x, width_ - validRoi_1.x - validRoi_1.width);
    int new_height = std::min(validRoi_1.height - (new_y_loc - validRoi_1.y), validRoi_2.height - (new_y_loc - validRoi_2.y));
    int new_width = std::min(std::min((validRoi_1.x + validRoi_1.width) - new_x_loc_right, validRoi_2.width), width_ - validRoi_2.x - new_x_loc_right);

    validMatchedRoi_1 = cv::Rect(width_ - new_x_loc_right - new_width, new_y_loc, new_width, new_height);
    validMatchedRoi_2 = cv::Rect(new_x_loc_right, new_y_loc, new_width, new_height);
}

