import numpy as np
import cv2
import pickle
import glob
#from tracker import tracker

import os.path

import utils

def undistort_test():
    calibration_pickle = pickle.load(open("./calibration_pickle.p", "rb"))
    mtx = calibration_pickle["mtx"]
    dist = calibration_pickle["dist"]

    img_paths = glob.glob('./test_images/test*.jpg')

    for fname in img_paths:
        img = cv2.imread(fname)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        saved_name = './test_images/' + 'undistorted_' + os.path.basename(fname)
        cv2.imwrite(saved_name, undistorted)
    # End of for fname
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread in BGR
    undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    utils.display_side_by_side(img, undistorted, caption="Undistorted_test",
                         save_path="./output_images")

def main():
    undistort_test()
    return 0

# imports
import sys
# constants

# exception classes

# interface functions

# classes

# internal functions & classes

if __name__ == '__main__':
    status = main()
    sys.exit(status)
