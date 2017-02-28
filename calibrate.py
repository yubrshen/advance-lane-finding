def calibrate():
    # Calibrate the camera, producing the calibration matrix

    # Prepare object points, perfect model
    import numpy as np
    import glob
    import cv2
    import pickle
    import os.path
    import utils

    # The number of inside corners horizontal, and vertical respectively
    nx, ny = 9, 6
    
    object_model = np.zeros((nx*ny, 3), np.float32)
    # coordinates of corners, (row, column, 0), there are nx*ny of such tuples.
    object_model[:,:2] = [[i, j] for j in range(ny) for i in range(nx)]
    # The above is equivalent to the following:
    # object_model[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # I don't understand the expression.
    
    # Store the object points and images points
    obj_points = []                 # for multiple object model compared to the image corner points
    img_points = []                 # for multiple image corner points
    
    # Get the list of images for calibration
    images = glob.glob("./camera_cal/calibration*.jpg")
    
    # process images for image corner points
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # find the corners on the chessboard
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        if ret == True:
            print("Working on ", fname)
            obj_points.append(object_model)
            img_points.append(corners)
        
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            saved_name = './output_images/' + 'corners_found_' + os.path.basename(fname)
            cv2.imwrite(saved_name, img)
        # End of if ret == True
    # End of for idx, fname
    
    img = cv2.imread("./camera_cal/calibration2.jpg")
    height, width = img.shape[:2] # the first two dimensions are the height, and width of the image
    
    # perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)
    
  
    # try to do undistort for the read img:
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    utils.display_side_by_side(img, dst, caption="Undistorted", save_path="./output_images")

    # store the calibration data in pickle
    calibration_pickle = {}
    calibration_pickle['mtx'] = mtx
    calibration_pickle['dist'] = dist
    pickle.dump(calibration_pickle, open("./calibration_pickle.p", "wb"))

    return mtx, dist

def main():
    calibrate()
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
