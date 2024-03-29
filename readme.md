# OpenCV Stereo Calibration using findChessboardCornersSB (live or offline)

Requirements:
```
OpenCV 4.5.1+  
Printed https://github.com/opencv/opencv/blob/master/modules/calib3d/doc/pics/checkerboard_radon.png
```



## Live Calibration
Uses cv::VideoCapture - the current string used assumes the camera feed is a UDP feed on port 5000 from GStreamer:   
e.g. on the computer with the cameras:
```
gst-launch-1.0 v4l2src ! video/x-raw, width=1280, height=400, format=GRAY8 ! videoconvert ! video/x-raw, format=I420 ! jpegenc ! image/jpeg ! jpegparse ! rtpjpegpay name=pay0 pt=96 ! udpsink port=5000 host=<ip of the computer running the calibration code>
```
## Offline Calibration
If doing offline calibration - see format from `generate_calib_xml.py`   
Place the xml file and pictures into the build/ folder   
e.g.
```
./single-camera-calibration leftcalib.xml -c=0
```
## Single camera calibration
To run the single camera calibration:  
1. Set the necessary parameters and check possible arguments in `single-camera-calibration/main.cpp`    
2. Compile e.g.
```
mkdir build
cd build
cmake ..
make
```
3. Run e.g. 
```
# if using the entire video feed
./single-camera-calibration -s=<size of one square in the chessboard in m. defaults to 0.01795>
# if using left half/right half of a stereo feed, set -c to 0 or 1 respectively: 
./single-camera-calibration -c=0 -s=<size of one square in the chessboard in m. defaults to 0.01795>
```
3. Follow instructions in the terminal (make sure camera feed is the foreground window when pressing hotkeys):  
Use spacebar to save the current image to use as a calibration image (needs to have the chessboard detected to accept it). Press enter after taking >10 images to run calibration. Press q or escape to exit program."; Usually at least 20 images is recommended.
4. Calibration parameters will be saved to `CalibParams_<image_side>.yml` by default

If running for a stereo camera calibration, make sure the -c flag is specified 

## Stereo camera calibration
This stereo calibration assumes you have already run single-camera-calibration on each individual camera to obtain their camera matrix and distortion coefficients. The resulting `CalibParams_0.yml` and `CalibParams_1.yml` from single-camera-calibration should be left in the build folder for use by the stereo calibrator.

Similar instructions as single camera calib   
Run:
```
./stereo-camera-calibration -s=<size of one square in the chessboard in m. defaults to 0.01795>
```


Unforunately, this calibrator requires the entire chessboard in the field of view of the camera. look at <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gadc5bcb05cb21cf1e50963df26986d7c9>?
