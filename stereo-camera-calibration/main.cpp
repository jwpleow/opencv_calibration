#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <sstream>

#include "OpenCVStereoCalibration.h"
#include "../CalibUtils.h"

/*
USAGE:
(perform single camera calibration first to get CalibParams_0.yml and CalibParams_1.yml and place them in the build folder for this)

stereo-camera-calibration -w=<board_width default=14> -h=<board_height default=9> -s=<square_size default=0.0174> <image list XML/YML file default=stereo_calib.xml>
Examples: 
`stereo-camera-calibration -w=14 -h=9 -s=0.0174 stereo_calib.xml` if calibration images are already taken
`stereo-camera-calibration -w=14 -h=9 -s=0.0174` if you want to use live video feed to take calibration images

if an image list file is provided, it should be in the format:
<?xml version="1.0"?>
<opencv_storage>
<imagelist>
"left-1.png"
"right-1.png"
</imagelist>
</opencv_storage>
(images are taken to be alternating between left and right)
*/

constexpr auto height = 400;
constexpr auto width = 1280; // width of original image, this should be twice of a single camera's width
const std::string output_fname = "CalibParams_Stereo.yml";
constexpr bool save_images = false; // whether to save calibration images taken if using live video feed
const int view_rectified = 5; // no of rectified images to view after calibrating
// calibration flags used in stereo_calib example - see flag details at stereoCalibrate() in OpenCV documentation
// const int calibration_flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_SAME_FOCAL_LENGTH + cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5;
const int calibration_flags = cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_INTRINSIC | cv::CALIB_SAME_FOCAL_LENGTH | cv::CALIB_TILTED_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_RATIONAL_MODEL;
int main(int argc, char **argv)
{
    std::string arg_keys = "{w|14|no. of squares in the x-direction of the chessboard}\
                {h|9|no. of squares in the y-direction of the chessboard}\
                {s|0.01795|size of each square on the chessboard(in metres)}\
                {@input||input xml file if using saved images}";
    cv::CommandLineParser parser(argc, argv, arg_keys);

    std::string image_list_file;
    const float chessboard_tile_size = parser.get<float>("s"); // size of each square on the chessboard (in metres)
    const cv::Size chessboard_size = cv::Size(parser.get<int>("w"), parser.get<int>("h"));
    if (parser.has("@input"))
    {
        image_list_file = parser.get<std::string>("@input");
    }
    if (!parser.check())
    {
        parser.printErrors();
        return false;
    }

    OpenCVStereoCalibration c(chessboard_size, chessboard_tile_size, calibration_flags, view_rectified);
    c.loadMonoCalibrationParams("CalibParams_0.yml", "CalibParams_1.yml");

    if (image_list_file.empty())
    {
        std::cout << "Connecting to VideoCapture device...\n";
        cv::VideoCapture vid_cap("udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink", cv::CAP_GSTREAMER);

        if (!vid_cap.isOpened())
        {
            std::cerr << "Video capture device could not be opened!";
            return 1;
        }
        std::cout << "Starting Calibration...\n";
        // vid_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        // vid_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        // system("v4l2-ctl --set-ctrl=gain=10");
        // system("v4l2-ctl --set-ctrl=exposure=1000");
        // system("v4l2-ctl --set-ctrl=gain=5");
        // system("v4l2-ctl --set-ctrl=exposure=300"); // needs to be set twice to work for some reason...

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
    { // using calibration images file
        std::cout << "Using calibration images file " << image_list_file << std::endl;
        std::vector<std::string> image_list;
        if (CalibUtils::readStringList(image_list_file, image_list))
        {
            c.loadImages(image_list);
            c.runStereoCalibration();
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
