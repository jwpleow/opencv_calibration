import os
import sys

## for offline single camera calibration
# reads the png files in a folder given in the argument and generates a file for them
# e.g. python generate_calib_xml.py single-camera-calibration/build/
print(f"Folder given: {sys.argv[1]}")
if not os.path.isdir(sys.argv[1]):
    sys.exit("Error: directory given is not valid")
files = os.listdir(sys.argv[1])
with open("left_calib.xml", 'w') as f:
    f.write(r'<?xml version="1.0"?>' + "\n")
    f.write(r"<opencv_storage>" + "\n")
    f.write(r"<imagelist>" + "\n")
    imgfiles = [filename for filename in sorted(files) if filename.endswith('.png')]
    print(f"found {len(imgfiles)} images")
    for imgfile in imgfiles:
        f.write('"' + imgfile + '"' + "\n")
    f.write(r"</imagelist>" + "\n")
    f.write(r"</opencv_storage>" + "\n")

## for offline stereo calibration
# generates stereo calib xml assuming images are "left-1.png", "right-1.png", and in sequential order
# number_to_generate_to = 63
# with open("stereo_calib.xml", 'w') as f:
#     f.write(r'<?xml version="1.0"?>' + "\n")
#     f.write(r"<opencv_storage>" + "\n")
#     f.write(r"<imagelist>" + "\n")
#     for i in range(1, number_to_generate_to + 1):
#         f.write(f'"left-{i}.png"' + "\n")
#         f.write(f'"right-{i}.png"' + "\n")
#     f.write(r"</imagelist>" + "\n")
#     f.write(r"</opencv_storage>" + "\n")
