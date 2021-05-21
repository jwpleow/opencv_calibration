#include "CalibUtils.h"
#include <ctime>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace CalibUtils{

std::string GetDateTime()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t); // not thread safe!
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

bool readStringList(const std::string &filename, std::vector<std::string> &l)
{
    l.resize(0);
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "file could not be opened!\n";
        return false;
    }
    cv::FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != cv::FileNode::SEQ)
    {
        std::cout << "node is not sequential type!\n";
        return false;
    }
    for (cv::FileNodeIterator it = n.begin(); it != n.end(); ++it)
        l.push_back(static_cast<std::string>(*it));
    return true;
}

void calculateKnownChessboardPositions(int num_images, cv::Size chessboard_size, float chessboard_tile_size, std::vector<std::vector<cv::Point3f>> &corners)
{
    corners.clear();
    corners.resize(1);
    corners[0].clear();
    for (int r = 0; r < chessboard_size.height; r++)
    {
        for (int c = 0; c < chessboard_size.width; c++)
        {
            corners[0].emplace_back(c * chessboard_tile_size, r * chessboard_tile_size, 0.0f);
        }
    }
    corners.resize(num_images, corners[0]); // duplicate to size num_images 
}

  double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>> &object_points,
                                                    const std::vector<std::vector<cv::Point2f>> &image_points,
                                                    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
    // taken from https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
    std::vector<cv::Point2f> image_points2; // to store the projected point output from projectPoints
    int i, total_points = 0;
    double total_err = 0, err;
    std::vector<float> per_view_errors;
    per_view_errors.resize(object_points.size());

    for (i = 0; i < static_cast<int>(object_points.size()); i++)
    {   
        cv::projectPoints(cv::Mat(object_points[i]), rvecs[i], tvecs[i], camera_matrix, dist_coeffs, image_points2);
        err = cv::norm(cv::Mat(image_points[i]), cv::Mat(image_points2), cv::NORM_L2);
        int n = static_cast<int>(object_points[i].size());
        per_view_errors[i] = static_cast<float>(std::sqrt(err * err / n));
        total_err += err * err;
        total_points += n;
    }
    return std::sqrt(total_err / total_points);
}

void getCalibrationFlagsString(int calibration_flags, std::string &calibration_flags_string)
{
    std::ostringstream oss;
    oss << (calibration_flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "");
    oss << (calibration_flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspect_ratio" : "");
    oss << (calibration_flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "");
    oss << (calibration_flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    oss << (calibration_flags & cv::CALIB_FIX_FOCAL_LENGTH ? "+fix_focal_length" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K1 ? "+fix_k1" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K2 ? "+fix_k2" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K3 ? "+fix_k3" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K4 ? "+fix_k4" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K5 ? "+fix_k5" : "");
    oss << (calibration_flags & cv::CALIB_FIX_K6 ? "+fix_k6" : "");
    oss << (calibration_flags & cv::CALIB_RATIONAL_MODEL ? "+rational_model" : "");
    oss << (calibration_flags & cv::CALIB_THIN_PRISM_MODEL ? "+thin_prism_model" : "");
    oss << (calibration_flags & cv::CALIB_TILTED_MODEL ? "+tilted_model" : "");
    oss << (calibration_flags & cv::CALIB_FIX_S1_S2_S3_S4 ? "+fix_s1_s2_s3_s4" : "");
    oss << (calibration_flags & cv::CALIB_FIX_TAUX_TAUY ? "+fix_taux_tauy" : "");
    oss << (calibration_flags & cv::CALIB_FIX_INTRINSIC ? "+fix_intrinsic" : "");
    oss << (calibration_flags & cv::CALIB_FIX_TANGENT_DIST ? "+fix_tangent_dist" : "");
    oss << (calibration_flags & cv::CALIB_SAME_FOCAL_LENGTH ? "+same_focal_length" : "");
    oss << (calibration_flags & cv::CALIB_ZERO_DISPARITY ? "+zero_disparity" : "");
    oss << (calibration_flags & cv::CALIB_USE_EXTRINSIC_GUESS ? "+use_extrinsic_guess" : "");
    calibration_flags_string = oss.str();
}

bool findCharucoBoardCorners(cv::Mat image, const cv::Ptr<cv::aruco::CharucoBoard> board, std::vector<cv::Point2f> charucoCorners, std::vector<int> charucoIds)
{
    /* from https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html
    
    If calibration parameters are provided, the ChArUco corners are interpolated by, first, estimating a rough pose from the 
    ArUco markers and, then, reprojecting the ChArUco corners back to the image.

    On the other hand, if calibration parameters are not provided, the ChArUco corners are interpolated by calculating the 
    corresponding homography between the ChArUco plane and the ChArUco image projection.

    The main problem of using homography is that the interpolation is more sensible to image distortion. Actually, the homography
    is only performed using the closest markers of each ChArUco corner to reduce the effect of distortion.

    When detecting markers for ChArUco boards, and specially when using homography, it is recommended to disable the corner 
    refinement of markers. The reason of this is that, due to the proximity of the chessboard squares, the subpixel process 
    can produce important deviations in the corner positions and these deviations are propagated to the ChArUco corner 
    interpolation, producing poor results. 
    
    The question is - should we then use initial single calibration camera matrix and dist coeffs to find refined corners during stereo calib?*/

    // cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    // cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE; // sample code uses this

    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedMarkers;
    std::vector<int> markerIds;
    cv::aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, detectorParams, rejectedMarkers);
    cv::aruco::refineDetectedMarkers(image, board, markerCorners, markerIds, rejectedMarkers); // is this bad?
    if (markerIds.size() > 4)
    {
        cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds);
        return true;
    }
    else
    {
        return false;
    }
}

} // namespace CalibUtils