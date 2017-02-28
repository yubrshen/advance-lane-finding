import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import os.path
import utils

def perspective_matrix_and_inverse():
    src = np.float32([(579,460),
                   (203,720),
                   (1127,720),
                   (711,460)])
    
    dst = np.float32([(350, 0),
                   (350,720),
                   (940,720),
                   (940,0)])
    lane_width_px = dst[3][0] - dst[0][0]
    lane_length_px = dst[1][1] - dst[0][1]
    xm_per_pix = 3.7/lane_width_px
    ym_per_pix = 30/lane_length_px 
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, xm_per_pix, ym_per_pix
def warp(img, M):
    height, width = img.shape[:2]
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

def warp_test():
    M, Minv, xm_per_pix, ym_per_pix = perspective_matrix_and_inverse()

    img_paths = glob.glob("./output_images/undistorted_*.jpg")

    for fname in img_paths:
        img = mpimg.imread(fname)
        warped = warp(img, M)

        saved_name = "./output_images/warped_" + os.path.basename(fname)

        mpimg.imsave(saved_name, warped)
    # End of for fname
    utils.display_side_by_side(img, warped, caption="warped", save_path="./output_images")

def main():
    warp_test()
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
