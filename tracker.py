def find_peak(histogram, x_left, x_right):
    """
    Given histogram, find the index of the peak in the interval on the histogram.
    Also provide confidence of the index for the peak, which is the ratio between
    the peak value and the mean of the histogram minus by one. """
    # Find the max from the histogram
    subset = histogram[int(x_left):int(x_right)]
    peak_index = np.argmax(subset)
    peak_value = subset[peak_index]
    mean_value = np.mean(subset)
    confidence = (peak_value - mean_value)/mean_value
    return (peak_index + x_left), confidence

prev_left_fit, prev_right_fit = None, None

def lanesDetection(warped, lane_centers):
      global prev_left_fit, prev_right_fit
      if (prev_left_fit is None) or (prev_right_fit is None):
            left_fit, right_fit, curverad, out_img, lane_quality = lane_centers.initial_lanes(warped)
            if lane_quality:
                  prev_left_fit, prev_right_fit = left_fit, right_fit
            # End of if lane_quality
      else:
            left_fit, right_fit, curverad, out_img, lane_quality = lane_centers.next_lanes(prev_left_fit, prev_right_fit, warped)
            if lane_quality:
                  prev_left_fit, prev_right_fit = left_fit, right_fit
            else:
                  prev_left_fit, prev_right_fit = None, None
            # End of if lane_quality
      # End of if prev_left_fit

      return out_img, left_fit, right_fit, curverad

def display_detected(self, left_fit, right_fit, window_width, img, Minv, curverad, lanes_on_warped):
    height, width = img.shape[:2]
    yvals = range(0, height)
    lane_detected = False
    if left_fit is not None:
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]  # populate all the points along vertical coordinate
        left_fitx = np.array(left_fitx, np.int32)
        self.recent_left_fitx.append((left_fitx))
        lane_detected = True
    else:
        left_fitx = None
    # End of if 0 < len(left_fit)

    if right_fit is not None:
        right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]  # populate all the points along vertical coordinate, to be continuous
        right_fitx = np.array(right_fitx, np.int32)
        self.recent_right_fitx.append((right_fitx))
        lane_detected = True
    else:
        right_fitx = None
    # End of if right_fit is not None

    if left_fitx is None:
        if 0 < len(self.recent_left_fitx):
            left_fitx = np.average(self.recent_left_fitx[-self.smooth_factor:], axis=0)
            lane_detected = True
        elif right_fitx is not None:
            left_fitx = right_fitx - self.estimated_lane_width_px
            lane_detected = True
        else:
            lane_detected = False
        # End of if right_fitx
    # End of if left_fitx is None

    if right_fitx is None:
        if 0 < len(self.recent_right_fitx):
            right_fitx = np.average(self.recent_right_fitx[-self.smooth_factor:], axis=0)
            lane_detected = True
        elif left_fitx is not None:
            right_fitx = left_fitx + self.estimated_lane_width_px
            lane_detected = True
        else:
            lane_detected = False
        # End of if left_fitx is not None
    # End of if right_fitx is not None

    if lane_detected:
        if self.debug:
            import matplotlib.pyplot as plt
            plt.imshow(lanes_on_warped)
            plt.plot(left_fitx, yvals, color='blue')
            plt.plot(right_fitx, yvals, color='blue')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig("./output_images/fitted_lines_on_warped.jpg")
            plt.show()
         # Endo of if debug

        # combine into extrapolated lane and marker image:
        left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1] + window_width/2), axis=0),
                                      np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.
        right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0),
                                       np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.

        center_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2, right_fitx[::-1] - window_width/2), axis=0),
                                        np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.
        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img)

        cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
        cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
        cv2.fillPoly(road, [center_lane], color=[0, 255, 0])
        cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

        update_lanes_on_warped = cv2.addWeighted(lanes_on_warped, 1.0, road, 0.3, 0.0)

        road_warped = cv2.warpPerspective(road, Minv, (width, height), flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (width, height), flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.3, 0.0)

        curverad_txt = str(round(curverad, 3)) if curverad else "unknown"
        cv2.putText(result,  curverad_txt + '(m) ' + 'Radius of Curvature' , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        lane_center_x, offset, position_side = lane_center(left_fitx, right_fitx, self.xm_per_pix, width)
        if offset == 0:
            off_center_txt = 'Right on Lane Center!'
        else:
            off_center_txt = str(abs(round(offset, 3))) + '(m) ' + position_side + ' Off Center'
        # End of if offset
        cv2.putText(result, off_center_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return result, offset, update_lanes_on_warped
    else:
        return img, None, img
    # End of if lane_detected

import numpy as np
import cv2
import matplotlib.pyplot as plt

class tracker():
    def __init__(self, My_ym =1, My_xm = 1, Mysmooth_factor = 15, debug=False):
        self.recent_centers = []
        self.recent_left_fit = []
        self.recent_right_fit = []
        self.recent_left_curverad = []
        self.recent_right_curverad = []
        # list that stores all the past (left, right) center set values used for smoothing the output
        self.recent_leftx_base = []
        self.recent_rightx_base = []
        self.recent_lane_width = []
        self.recent_left_fitx = []
        self.recent_right_fitx = []
        self.xm_per_pix = My_xm  # meters per pixel in horizontal axis
        self.ym_per_pix = My_ym  # meters per pixel in vertical axis
        self.smooth_factor = Mysmooth_factor
        self.debug = debug
        self.estimated_lane_width_meter = 3.7
        self.minimum_lane_width_px = 3.5/self.xm_per_pix
        self.maxmum_lane_width_px = 3.8/self.xm_per_pix
        self.estimated_lane_width_px = 3.7/self.xm_per_pix
        
    def initial_lane_start_x_positions(self, binary_warped):
         # Find the peak of the left and right halves of the histogram
         # These will be the starting point for the left and right lines
         height, width = binary_warped.shape[:2]
         scan_start = 150
         midpoint = np.int(width/2)
         scan_end = width
         top_exclusison = 0.6
         y_window_top = np.int(height*top_exclusison)
         y_window_bottom = np.int(height*0.95)
         histogram = np.sum(binary_warped[y_window_top:y_window_bottom,:], axis=0)
    
         leftx_base, leftx_base_confidence = find_peak(histogram, scan_start, midpoint)
         rightx_base, rightx_base_confidence = find_peak(histogram, midpoint, scan_end)
    
         self.recent_leftx_base.append(leftx_base)
         self.recent_rightx_base.append(rightx_base)
    
         # assuming the two starting are the lane start, their distance should be the road width
         estimated_width = rightx_base - leftx_base
         estimated_width = max(estimated_width, self.minimum_lane_width_px)
         estimated_width = min(estimated_width, self.maxmum_lane_width_px)
         self.recent_lane_width.append(estimated_width)
    
         avg_leftx_base = int(np.average(self.recent_leftx_base[-self.smooth_factor:], axis=0))
         avg_rightx_base = int(np.average(self.recent_rightx_base[-self.smooth_factor:], axis=0))
         self.estimated_lane_width_px = int(np.average(self.recent_lane_width[-self.smooth_factor:], axis=0))
    
         return (avg_leftx_base if (leftx_base_confidence < 0.01) else leftx_base,
                 avg_rightx_base if (rightx_base_confidence < 0.01) else rightx_base)
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    def initial_lanes(self, binary_warped):
        """
        Starting from the bottom of the image,
        using the starting x-coordinates of left and right lanes as seeds to search nonzero points
        in the neighborhood of the seeds, and
        consider them to be part of the lanes, respectively for the left, and right
        update the seeds based on newly found nonzero points, and continue the search upward.
        Use the lane indices found to produce fitted 2nd order polynomial for the left lane,
        and right lane, and the curvature.
        """
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Choose the number of sliding windows
        nwindows = 9 # 18 # 36 # 9
        left_confident_credit = nwindows
        right_confident_credit = nwindows
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin_initial = 100 # 70
        margin = {}
        margin['left'] = margin_initial
        margin['right'] = margin_initial
        # Set minimum number of pixels found to recenter window
        minpix = 50 # 2 # 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Current positions to be updated for each window
        leftx_current, rightx_current = self.initial_lane_start_x_positions(binary_warped)
        def lane_indices(i, starting_x, margin_side):            # i-th windows
            # Identify window boundaries in x and y (and right and left)
            win_y_low = max(0, np.int(binary_warped.shape[0]*0.98) - (i+1)*window_height)  # ignore the bottom part, which is noise.
            win_y_high = max(0, np.int(binary_warped.shape[0]*0.98) - i*window_height)
    
            margin[margin_side] = int(margin[margin_side]*(1 + 0.005))  # adaptive to enlarge search
            win_x_low = starting_x - margin[margin_side]
            win_x_high = starting_x + margin[margin_side]
    
            # Identify the nonzero pixels in x and y within the window
            # try to search towards the center of the lane first
            if margin_side == 'left':
                priority_lane_inds = ((win_y_low <= nonzeroy ) & (nonzeroy < win_y_high) &
                             (starting_x <= nonzerox ) & (nonzerox < win_x_high)).nonzero()[0]
                if minpix <= len(priority_lane_inds):
                    cv2.rectangle(out_img,(starting_x, win_y_low), (win_x_high, win_y_high), (0,255,0), 2)
                    confident = True
                    return priority_lane_inds, confident
                else:
                    remaining_inds = ((win_y_low <= nonzeroy ) & (nonzeroy < win_y_high) &
                                      (win_x_low <= nonzerox ) & (nonzerox < starting_x)).nonzero()[0]
                    lane_inds = np.concatenate((priority_lane_inds, remaining_inds))
                # End of if minpix < ...
            elif margin_side == 'right':
                priority_lane_inds = ((win_y_low <= nonzeroy ) & (nonzeroy < win_y_high) &
                             (win_x_low <= nonzerox ) & (nonzerox < starting_x)).nonzero()[0]
                if minpix <= len(priority_lane_inds):
                    cv2.rectangle(out_img,(win_x_low, win_y_low), (starting_x, win_y_high), (0,255,0), 2)
                    confident = True
                    return priority_lane_inds, confident
                else:
                    remaining_inds = ((win_y_low <= nonzeroy ) & (nonzeroy < win_y_high) &
                                      (starting_x <= nonzerox ) & (nonzerox < win_x_high)).nonzero()[0]
                    lane_inds = np.concatenate((priority_lane_inds, remaining_inds))
                # End of if minpix < ...
            # End of if margin_side == ...
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_low, win_y_low), (win_x_high,win_y_high), (0,255,0), 2)
            confident = ((minpix + 0) < len(lane_inds)) # relax the criteria for confidence
            return lane_inds, confident
    
        # Step through the windows one by one
        for window in range(nwindows):
            good_left_inds, confident = lane_indices(window, leftx_current, 'left')
            # If you found more than minpix pixels, recenter next window on their mean position
            left_lane_inds.append(good_left_inds)
            if confident:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if right_confident_credit < nwindows*0.5:
                    rightx_current = np.int(np.mean(np.array([rightx_current, 
                                                              leftx_current + self.estimated_lane_width_px])))
                # consider the left lane is often more reliable
                margin['left'] = margin_initial  # reset
            else:
                left_confident_credit = left_confident_credit - 1
            good_right_inds, confident = lane_indices(window, rightx_current, 'right')
            right_lane_inds.append(good_right_inds)
            # If you found more than minpix pixels, recenter next window on their mean position
            if confident:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                margin['right'] = margin_initial  # reset
            else:
                right_confident_credit = right_confident_credit - 1
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        left_fit, left_curverad, right_fit, right_curverad, lane_data_quality = self.lane_poly_fit_and_curveture(
            nonzerox, nonzeroy, left_lane_inds, right_lane_inds, np.float(left_confident_credit/nwindows), np.float(right_confident_credit/nwindows))
        return left_fit, right_fit, left_curverad, out_img, lane_data_quality
    def next_lanes(self, prev_left_fit, prev_right_fit, binary_warped):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        """
        Return lanes as polynomials fit, based on binary_warped, and the previous lanes fit
        """
        nonzero = binary_warped.nonzero()  # the tuple of indices for x, and y, in binary_warped that have nonezero values
        nonzeroy = np.array(nonzero[0]) # the x indices for nonzero pixels
        nonzerox = np.array(nonzero[1])  # the y indices for nonzero pixels
        margin = 100 # defines the neighborhood to find the lane from the previous lane. (The previous lane is defined by a 2nd order polynomial)
    
        def next_lane_indices(prev_fit):
            previous_x_inds = (prev_fit[0]*(nonzeroy**2) + prev_fit[1]*nonzeroy + prev_fit[2])
            # based on the current nozeroyy, expecting the current nonzerox should be close enough
            lane_inds = ((nonzerox > (previous_x_inds - margin)) & (nonzerox < (previous_x_inds + margin)))
            # The range of x-indices for the current lane, based on previous lane's y-indices.
            return lane_inds
    
        left_lane_inds = next_lane_indices(prev_left_fit)
        right_lane_inds = next_lane_indices(prev_right_fit)
    
        left_fit, left_curverad, right_fit, right_curverad, lane_data_quality = self.lane_poly_fit_and_curveture(
            nonzerox, nonzeroy, left_lane_inds, right_lane_inds, 1, 1)
    
        # The following code is just to visualize, not for function. 
        # Generate x and y values for plotting, for the lines fit
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # draw the identified left lane
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        if self.debug:
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
        # Endo of if debug
        # The end of visualization
    
        return left_fit, right_fit, left_curverad, result, lane_data_quality
    def lane_poly_fit_and_curveture(self, nonzerox, nonzeroy, left_lane_inds, right_lane_inds,
                                    left_confidence_ratio, right_confidence_ratio):
          """
           produces the fitted polynomials for the left, and right lane, and the curvature of the left lane. 
           It takes the indices of the left lane, and right lane found by initial_lanes, and next_lanes.
           """
          left_data_num = left_lane_inds.size # collect data
          right_data_num = right_lane_inds.size # collect data
    
          left_x, left_y, right_x, right_y = [], [], [], []
          # Extract line pixel positions
          if (0  < left_data_num):
                left_x = nonzerox[left_lane_inds] 
                left_y = nonzeroy[left_lane_inds] 
          # End of if  
          if (0  < right_data_num):
                right_x = nonzerox[right_lane_inds]
                right_y = nonzeroy[right_lane_inds] 
          # End of if 
    
          # Augment the data:
          if (left_confidence_ratio < 0.4) and (0.5 < right_confidence_ratio):
                left_y = np.concatenate((left_y, right_y))
                left_x = np.concatenate((left_x, right_x - self.estimated_lane_width_px))
          # End of if (left_concatenate_ratio...)
          if (right_confidence_ratio < 0.4) and (0.5 < left_confidence_ratio):
                right_y = np.concatenate((right_y, left_y))
                right_x = np.concatenate((right_x, left_x + self.estimated_lane_width_px))
          # End of if
    
          def lane_fit(y, x):
                # Fit a second order polynomial
                fit = np.polyfit(y, x, 2)
                x_m = x*self.xm_per_pix
                y_m = y*self.ym_per_pix
                lane_fit_meter = np.polyfit(y_m, x_m, 2)
                curverad = ((1 + (2*lane_fit_meter[0]*y_m[-1] + lane_fit_meter[1])**2)**1.5)/np.absolute(2*lane_fit_meter[0])
    
                return fit, curverad
          def history_average(x, label=""):
                print('No valid ' + label + ' found. Retrieve from history average.')
                if 0 < len(x):
                      result = np.average(x[-self.smooth_factor:], axis=0)
                else:
                      result = None
                      print('No average from history found, ' + label)
                # End of if 0 < len(x)
                return result
    
          if (0 < left_data_num) and ((0.05 < left_confidence_ratio) or (self.recent_left_fit == [])):
                left_lane_fit, left_curverad = lane_fit(left_y, left_x)
                self.recent_left_fit.append(left_lane_fit)
                self.recent_left_curverad.append(left_curverad)
          else:
                left_lane_fit = history_average(self.recent_left_fit, label="left lane")
                left_curverad = history_average(self.recent_left_curverad, label="left curverad")
          # End of if
    
          if (0 < right_data_num) and ((0.05 < right_confidence_ratio) or (self.recent_right_fit == [])):
                right_lane_fit, right_curverad = lane_fit(right_y, right_x)
                self.recent_right_fit.append(right_lane_fit)
                self.recent_right_curverad.append(right_curverad)
          else:
                right_lane_fit = history_average(self.recent_right_fit, label="right_lane")
                right_curverad = history_average(self.recent_right_curverad, label="right curverad")
          # End of if
          return left_lane_fit, left_curverad, right_lane_fit, right_curverad, ((0.6 < left_confidence_ratio) or (0.6 < right_confidence_ratio))
    def display_detected(self, left_fit, right_fit, window_width, img, Minv, curverad, lanes_on_warped):
        height, width = img.shape[:2]
        yvals = range(0, height)
        lane_detected = False
        if left_fit is not None:
            left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]  # populate all the points along vertical coordinate
            left_fitx = np.array(left_fitx, np.int32)
            self.recent_left_fitx.append((left_fitx))
            lane_detected = True
        else:
            left_fitx = None
        # End of if 0 < len(left_fit)
    
        if right_fit is not None:
            right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]  # populate all the points along vertical coordinate, to be continuous
            right_fitx = np.array(right_fitx, np.int32)
            self.recent_right_fitx.append((right_fitx))
            lane_detected = True
        else:
            right_fitx = None
        # End of if right_fit is not None
    
        if left_fitx is None:
            if 0 < len(self.recent_left_fitx):
                left_fitx = np.average(self.recent_left_fitx[-self.smooth_factor:], axis=0)
                lane_detected = True
            elif right_fitx is not None:
                left_fitx = right_fitx - self.estimated_lane_width_px
                lane_detected = True
            else:
                lane_detected = False
            # End of if right_fitx
        # End of if left_fitx is None
    
        if right_fitx is None:
            if 0 < len(self.recent_right_fitx):
                right_fitx = np.average(self.recent_right_fitx[-self.smooth_factor:], axis=0)
                lane_detected = True
            elif left_fitx is not None:
                right_fitx = left_fitx + self.estimated_lane_width_px
                lane_detected = True
            else:
                lane_detected = False
            # End of if left_fitx is not None
        # End of if right_fitx is not None
    
        if lane_detected:
            if self.debug:
                import matplotlib.pyplot as plt
                plt.imshow(lanes_on_warped)
                plt.plot(left_fitx, yvals, color='blue')
                plt.plot(right_fitx, yvals, color='blue')
                plt.xlim(0, 1280)
                plt.ylim(720, 0)
                plt.savefig("./output_images/fitted_lines_on_warped.jpg")
                plt.show()
             # Endo of if debug
    
            # combine into extrapolated lane and marker image:
            left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1] + window_width/2), axis=0),
                                          np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.
            right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0),
                                           np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.
    
            center_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2, right_fitx[::-1] - window_width/2), axis=0),
                                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)  # the line of the lane's left edge downward, and the lane's right edged upward.
            road = np.zeros_like(img)
            road_bkg = np.zeros_like(img)
    
            cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
            cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
            cv2.fillPoly(road, [center_lane], color=[0, 255, 0])
            cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
            cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])
    
            update_lanes_on_warped = cv2.addWeighted(lanes_on_warped, 1.0, road, 0.3, 0.0)
    
            road_warped = cv2.warpPerspective(road, Minv, (width, height), flags=cv2.INTER_LINEAR)
            road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (width, height), flags=cv2.INTER_LINEAR)
    
            base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
            result = cv2.addWeighted(base, 1.0, road_warped, 0.3, 0.0)
    
            curverad_txt = str(round(curverad, 3)) if curverad else "unknown"
            cv2.putText(result,  curverad_txt + '(m) ' + 'Radius of Curvature' , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
            lane_center_x, offset, position_side = lane_center(left_fitx, right_fitx, self.xm_per_pix, width)
            if offset == 0:
                off_center_txt = 'Right on Lane Center!'
            else:
                off_center_txt = str(abs(round(offset, 3))) + '(m) ' + position_side + ' Off Center'
            # End of if offset
            cv2.putText(result, off_center_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return result, offset, update_lanes_on_warped
        else:
            return img, None, img
        # End of if lane_detected
    # End class tracker

def lane_center(left_fitx, right_fitx, xm_per_pix, width):
    lane_center_x = (left_fitx[-1] + right_fitx[-1])/2
    offset = (-lane_center_x + width/2)*xm_per_pix
    position_side = 'Right' if 0 < offset else 'Left'
    return lane_center_x, offset, position_side
