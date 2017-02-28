import utils
import cv2
import matplotlib.image as mpimg
import numpy as np

def binary_for_lanes(image, fname=None, debug=False):
    import numpy as np
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    binary_mask = get_lane_lines_mask(hsv_image, [WHITE_LINES, YELLOW_LINES])
    if debug:
        titles = [fname, 'HSV', 'White and Yellow Color Extracted']
        images = [image, hsv_image, binary_mask]
        cmaps = [None, None, 'gray']
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1,3,i+1), plt.imshow(images[i], cmap=cmaps[i])
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        # End of for i...
        import os.path
        plt.savefig("./output_images/binary_study_" + os.path.basename(fname))
        plt.show()
    # End of if debug
    return binary_mask
def gimp_to_opencv_hsv(*rgb):
    """
        from GIMP color values (RGB) to HSV
        """
    return (rgb[0] / 2, rgb[1] / 100 * 255, rgb[2] / 100 * 255)
      
# White and yellow color thresholds for lines masking.
# Optional "kernel" key is used for additional morphology
WHITE_LINES = { 'low_th': gimp_to_opencv_hsv(0, 0, 80),
                'high_th': gimp_to_opencv_hsv(359, 10, 100) }

YELLOW_LINES = { 'low_th': gimp_to_opencv_hsv(35, 20, 30),
                 'high_th': gimp_to_opencv_hsv(65, 100, 100),
                 'kernel': np.ones((3,3),np.uint64)}

def get_lane_lines_mask(hsv_image, colors):
    """
        Image binarization using a list of colors. The result is a binary mask
        which is a sum of binary masks for each color.
        """
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
            if 'kernel' in color:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
            # End of if 'kernel' ...
            masks.append(mask)
        else: raise Exception('High or low threshold values missing')
        # End of if 'low_th' ...
    # End of for color ...
    if masks:
        return cv2.add(*masks)

import utils
def to_RGB(img):
   if img.ndim == 2:
       return np.dstack((img, img, img))
   elif img.ndim == 3:
       return img
   else:
       return None

def compose_diagScreen(curverad=0, offset=0, mainDiagScreen=None,
                     diag1=None, diag2=None, diag3=None, diag4=None, diag5=None, diag6=None, diag7=None, diag8=None, diag9=None):
      # middle panel text example
      # using cv2 for drawing text in diagnostic pipeline.
      font = cv2.FONT_HERSHEY_COMPLEX
      middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
      cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(curverad), (30, 60), font, 1, (255,0,0), 2)
      cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(offset), (30, 90), font, 1, (255,0,0), 2)

      # assemble the screen example
      diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
      if mainDiagScreen is not None:
            diagScreen[0:720, 0:1280] = mainDiagScreen
      if diag1 is not None:
            diagScreen[0:240, 1280:1600] = cv2.resize(to_RGB(diag1), (320,240), interpolation=cv2.INTER_AREA) 
      if diag2 is not None:
            diagScreen[0:240, 1600:1920] = cv2.resize(to_RGB(diag2), (320,240), interpolation=cv2.INTER_AREA)
      if diag3 is not None:
            diagScreen[240:480, 1280:1600] = cv2.resize(to_RGB(diag3), (320,240), interpolation=cv2.INTER_AREA)
      if diag4 is not None:
            diagScreen[240:480, 1600:1920] = cv2.resize(to_RGB(diag4), (320,240), interpolation=cv2.INTER_AREA)*4
      if diag7 is not None:
            diagScreen[600:1080, 1280:1920] = cv2.resize(to_RGB(diag7), (640,480), interpolation=cv2.INTER_AREA)*4
      diagScreen[720:840, 0:1280] = middlepanel
      if diag5 is not None:
            diagScreen[840:1080, 0:320] = cv2.resize(to_RGB(diag5), (320,240), interpolation=cv2.INTER_AREA)
      if diag6 is not None:
            diagScreen[840:1080, 320:640] = cv2.resize(to_RGB(diag6), (320,240), interpolation=cv2.INTER_AREA)
      if diag9 is not None:
            diagScreen[840:1080, 640:960] = cv2.resize(to_RGB(diag9), (320,240), interpolation=cv2.INTER_AREA)
      if diag8 is not None:
            diagScreen[840:1080, 960:1280] = cv2.resize(to_RGB(diag8), (320,240), interpolation=cv2.INTER_AREA)

      return diagScreen

import numpy as np
import glob
import cv2
import pickle
import os.path
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warp
import tracker

def test_pipeline():
      calibration_pickle = pickle.load(open("./calibration_pickle.p", "rb"))
      mtx = calibration_pickle["mtx"]
      dist = calibration_pickle["dist"]
      M, Minv, xm_per_pix, ym_per_pix = warp.perspective_matrix_and_inverse()
      img_paths = glob.glob('./test_images/*.jpg')
      lane_centers = tracker.tracker(My_ym=ym_per_pix, My_xm=xm_per_pix,
                                     Mysmooth_factor=1, debug=True)
      for fname in img_paths:
            print('working on ', fname)
            img = mpimg.imread(fname)
            result, binary, warped, undistorted = pipeline(img, mtx, dist, M, Minv, lane_centers, debug=True, raw_fname=fname) 
            saved_name = "./output_images/lane_center_" + os.path.basename(fname)
            mpimg.imsave(saved_name, result)
      # End of for fname
      # illustrate the perspective transform. 
      utils.display_side_by_side(undistorted, result, caption="lane_center_test", cmap='gray',
                                 save_path="./output_images")

from crop import keep_region_of_interest
def pipeline(img, mtx, dist, M, Minv, lane_centers, debug=False, raw_fname=None):
      # the pipeline to process images.
      undistorted = cv2.undistort(img, mtx, dist, None, mtx)

      binary = binary_for_lanes(undistorted)
      # remove distraction by cropping to keep on the region of interests
      cropped = keep_region_of_interest(binary)
      # cropped = binary
      if debug:
            mpimg.imsave("./output_images/binary_"+ os.path.basename(raw_fname), cropped, cmap='gray')

      warped = warp.warp(cropped, M)
      if debug:
            saved_name = './output_images/' + 'warped_' + os.path.basename(raw_fname)
            mpimg.imsave(saved_name, warped, cmap='gray')

      lanes_on_warped, left_fit, right_fit, curverad = tracker.lanesDetection(warped, lane_centers)
      result, offset, lanes_on_warped = lane_centers.display_detected(
            left_fit, right_fit, 5, img, Minv, curverad, lanes_on_warped)
      result = compose_diagScreen(curverad=curverad, offset=offset, mainDiagScreen=result,
                                  diag1 = binary, diag2 = cropped, 
                                  diag4=warped, diag5 = undistorted,  
                                  diag7=lanes_on_warped)
      return result, binary, warped, undistorted

def main():
      test_pipeline()
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
