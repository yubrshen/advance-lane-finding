import pipeline

from moviepy.editor import VideoFileClip
import glob
import pickle
import os.path
import warp
import matplotlib.pyplot as plt
import tracker

def main():
    calibration_pickle = pickle.load(open("./calibration_pickle.p", "rb"))
    mtx = calibration_pickle["mtx"]
    dist = calibration_pickle["dist"]
    M, Minv, xm_per_pix, ym_per_pix = warp.perspective_matrix_and_inverse()
    lane_centers = tracker.tracker(My_ym=ym_per_pix, My_xm=xm_per_pix,
                                   Mysmooth_factor=5)

    def mark_lane_frame(img):
        result, binary, warped, undistorted = pipeline.pipeline(img, mtx, dist, M, Minv, lane_centers)
        return result

    video_paths = glob.glob("./*.mp4")

    for v_p in video_paths:
        clip = VideoFileClip(v_p)
        output_clip = clip.fl_image(mark_lane_frame)
        output_clip.write_videofile("./output_images/marked_"+os.path.basename(v_p), audio=False)
    # End of for v_p

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
