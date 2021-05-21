#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <thread>

#include "../CalibUtils.h"
#include "OpenCVCalibration.h"

/*
example:
./camera_calibration -c=1                    => use right side camera
./camera_calibration left_calib_images.xml   => use images in the xml file for calibration (And assumes c=0 => saves to CalibParams0.txt)
*/

constexpr auto height = 400;
constexpr auto width = 1280; // width of original image, this should be twice of a single camera's width if using image_side != -1
constexpr bool save_images = false; // whether to save images captured. only used if doing live calibration
const int calibration_flags = cv::CALIB_TILTED_MODEL | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_RATIONAL_MODEL;
// const int calibration_flags = 0;
// opencv calib example's flags
// const int calibration_flags = cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT;
constexpr int view_undistorted = 3; // how many undistorted versions of the calibration images to view after calibration
int main(int argc, char** argv)
{
    /* 
    arducam's gain and exposure can be set via v4l2-ctl --set-ctrl=gain=5
    check value using v4l2-ctrl --all
    but must be called after opening video capture
    */
    std::string arg_keys = "{w|14|no. of squares in the x-direction of the chessboard}\
                {h|9|no. of squares in the y-direction of the chessboard}\
                {s|0.01795|size of each square on the chessboard(in metres)}\
                {c|-1|side of the video feed / which camera this is for. 0 to use the left-half, 1 for right-half, -1 to use the whole image. }\
                {@input||input xml file if using saved images}";
    cv::CommandLineParser parser(argc, argv, arg_keys);

    std::string image_list_file;
    const float chessboard_tile_size = parser.get<float>("s"); // size of each square on the chessboard (in metres)
    const cv::Size chessboard_size = cv::Size(parser.get<int>("w"), parser.get<int>("h"));
    const int image_side = parser.get<int>("c");
    if (parser.has("@input"))
    {
        image_list_file = parser.get<std::string>("@input");
    }
    if (!parser.check())
    {
        parser.printErrors();
        return false;
    }

    std::ostringstream oss;
    oss << "CalibParams_" << image_side << ".yml"; // output file name for the calibration parameters
    const std::string output_fname = oss.str();
    OpenCVCalibration c(chessboard_size, chessboard_tile_size, image_side, calibration_flags, view_undistorted);
    if (image_list_file.empty())
    {
        cv::VideoCapture vid_cap("udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink", cv::CAP_GSTREAMER);

        if (!vid_cap.isOpened())
        {
            std::cerr << "Video capture device could not be opened!";
            return 0;
        }

        // vid_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        // vid_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        std::this_thread::sleep_for(std::chrono::seconds(2));
        // system("v4l2-ctl --set-ctrl=gain=10");
        // system("v4l2-ctl --set-ctrl=exposure=1000");
        // system("v4l2-ctl --set-ctrl=gain=5");
        // system("v4l2-ctl --set-ctrl=exposure=300"); // needs to be set twice to work for some reason?
        c.startCalibrationLoop(vid_cap, save_images);
        if (c.isCalibrated())
        {
            c.saveCalibrationParams(output_fname);
        }
        else
        {
            std::cout << "Calibration was unsuccessful.\n";
        }
    }
    else
    {
        std::cout << "Using calibration images file " << image_list_file << std::endl;
        std::vector<std::string> image_list;
        if (CalibUtils::readStringList(image_list_file, image_list))
        {
            c.loadImages(image_list);
            c.runCalibration();
            if (c.isCalibrated())
            {
                c.saveCalibrationParams(output_fname);
            }
        }
        else
        {
            std::cout << "Could not read input string list!\n";
            return 1;
        }
    }
    

    return 0;
}