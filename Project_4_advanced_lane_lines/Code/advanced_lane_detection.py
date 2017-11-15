import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

class Line():
    def __init__(self, buffer_length=10):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array(False)]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def update(self, new_x, new_y, coefficient, detected_flag):
        self.detected = detected_flag
        self.recent_xfitted.append(new_x)
        self.current_fit = coefficient
        self.allx = new_x
        self.ally = new_y
        self.radius_of_curvature = self.curvature_estimate(y_eval=720)

    def curvature_estimate(self, y_eval):

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        return curverad

def load_params(calibration_params, top_down_params):

    # load calibration parameters
    with open(calibration_params, 'rb') as pickle_file:
        calibration_data = pickle.load(pickle_file)
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']

    # load perspective transform parameters
    with open(top_down_params, 'rb') as pickle_file:
        perspective_data = pickle.load(pickle_file)
    M = perspective_data['M']
    Minv = perspective_data['Minv']

    return mtx, dist, M, Minv

def preprocess(img_undist, verbose=False):
    '''
        Thresholding the input image to generate a binary image by
        combining color and gradient information
        :param img:
        :return: binary image
    '''

    # sobel x
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(gray)
    sxbinary[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1

    # color
    hls = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_channel_binary = np.zeros_like(gray)
    s_channel_binary[(s_channel > s_thresh_min) & (s_channel < s_thresh_max)] = 1

    # combine the two binary thresholds
    combine_binary = np.zeros_like(gray)
    combine_binary[(sxbinary == 1) | (s_channel_binary == 1)] = 1

    if verbose:
        # plot
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(40, 40))
        ax1.set_title('undistorted image')
        ax1.imshow(img_undist)

        ax2.set_title('color thresholding')
        ax2.imshow(s_channel_binary, cmap='gray')

        ax3.set_title('x gradient thresholding')
        ax3.imshow(sxbinary, cmap='gray')

        ax4.set_title('combined binary')
        ax4.imshow(combine_binary, cmap='gray')

    return combine_binary

def sanity_check(left_line, right_line):
    diff_curvature = np.absolute(left_line.radius_of_curvature - right_line.radius_of_curvature)

    horizaontal_distance = []
    for y_value in range(img_size[1]):
        left_point_x = left_line.current_fit[0] * y_value ** 2 + left_line.current_fit[1] * y_value + \
                       left_line.current_fit[2]
        right_point_x = right_line.current_fit[0] * y_value ** 2 + right_line.current_fit[1] * y_value + \
                        right_line.current_fit[2]
        horizaontal_distance.append(right_point_x - left_point_x)
    distance_mean = np.mean(horizaontal_distance)
    distance_std = np.std(horizaontal_distance)

    if (diff_curvature < 1000) and (distance_mean * 3.7 / 700 > 1.5) and (distance_std * 3.7 / 700 < 0.5):
        ret = True
    else:
        ret = False
    return ret

def poly_fit(binary_TD, verbose=False):
    '''
        fit a polynomial by given warped binary image
        :param binary_TD: binary top-down view image
        :return: polynomial coefficient
    '''
    # take a histogram of the bottom half of the image
    histogram = np.sum(binary_TD[binary_TD.shape[0]//2:, :], axis=0)
    # create an output image to draw on and visualize the result
    out_img = np.dstack((binary_TD, binary_TD, binary_TD)) * 255
    # starting point for the left and right lines
    midpoint= np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # choose the number of sliding windows
    nWindows = 9
    # set height of windows
    window_height = np.int(binary_TD.shape[0]/nWindows)
    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_TD.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # set the width of the windows +/- margin
    margin = 100
    # set minimum number of pixels found to recenter window
    minpix = 50
    # create empty lists to receive left and right lane pixels indices
    left_lane_inds = []
    right_lane_inds = []

    # step through the windows one by one
    for window in range(nWindows):
        # identify window boundaries in x and y (and left and right)
        win_y_low = binary_TD.shape[0] - (window + 1) * window_height
        win_y_high = binary_TD.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # draw the windows on the visulization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]
        # append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.update(leftx, lefty, left_fit, detected_flag=True)
    right_line.update(rightx, righty, right_fit, detected_flag=True)

    if verbose:
        # generate x and y values for plotting
        ploty = np.linspace(0, binary_TD.shape[0]-1, binary_TD.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure('Sliding Windows Polyfit')
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return out_img, left_line, right_line

def fast_poly_fit(binary_TD, left_line, right_line, verbose=False):
    '''
        Assume you now have a new warped binary image
        from the next frame of video (also called "binary_TD")
        It's now much easier to find line pixels
    '''
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    nonzero = binary_TD.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.update(leftx, lefty, left_fit, detected_flag=True)
    right_line.update(rightx, righty, right_fit, detected_flag=True)

    if verbose:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_TD.shape[0] - 1, binary_TD.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_TD, binary_TD, binary_TD)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure("Fast Polyfit Search Aera")
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return result, left_line, right_line

def remap(binary_TD, left_fit, right_fit, Minv):

    img_size = (1280, 720)

    # generate x and y values for plotting
    ploty = np.linspace(0, binary_TD.shape[0] - 1, binary_TD.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_TD).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)

    return newwarp

def visulization(result, binary, binary_TD, disp_img, left_line, right_line, offset):
    h, w = result.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = result.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    result = cv2.addWeighted(src1=mask, alpha=0.2, src2=result, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    result[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(binary_TD, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    result[off_y:thumb_h + off_y, 2 * off_x + thumb_w:2 * (off_x + thumb_w), :] = thumb_birdeye

    thumb_disp_img = cv2.resize(disp_img, dsize=(thumb_w, thumb_h))
    result[off_y:thumb_h + off_y, 3 * off_x + 2 * thumb_w:3 * (off_x + thumb_w), :] = thumb_disp_img

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([left_line.radius_of_curvature, right_line.radius_of_curvature])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Offset from center: {:.02f}m'.format(offset), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def process_pipeline(frame, keep_state=True):

    global left_line, right_line, processed_frame

    img_undist = cv2.undistort(frame, mtx, dist, None, mtx)
    binary = preprocess(img_undist)
    binary_TD = cv2.warpPerspective(binary, M, img_size)

    # fit polynomial lines
    if processed_frame == 0 or not sanity_check(left_line, right_line):
        #print('/nre-polyfiting ...')
        disp_img, left_line, right_line = poly_fit(binary_TD, verbose=True)
    else:
        disp_img, left_line, right_line = fast_poly_fit(binary_TD, left_line, right_line, verbose=True)

    # Center of two lines
    y_bottom = binary_TD.shape[0]
    left_point_x = left_line.current_fit[0]*y_bottom**2 + left_line.current_fit[1]*y_bottom + left_line.current_fit[2]
    right_point_x = right_line.current_fit[0]*y_bottom**2 + right_line.current_fit[1]*y_bottom + right_line.current_fit[2]
    lane_center = np.float32((right_point_x + left_point_x) / 2)

    # Offset of vehicle's center
    offset = (binary_TD.shape[1]/2 - lane_center) * (3.7 / 700)  # time meters per pixel in x axis

    newwarp = remap(binary_TD, left_line.current_fit, right_line.current_fit, Minv)
    result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
    visul_result = visulization(result, binary, binary_TD, disp_img, left_line, right_line, offset)
    processed_frame += 1

    return visul_result

if __name__ == '__main__':

    # load calibration and perspective transform parameters
    mtx, dist, M, Minv = load_params('calibration_params.pkl', 'top_down_params.pkl')
    img_size = (1280, 720)

    # Mode selection
    mode = 'video'
    filename = '../project_video.mp4'

    # Define two lines
    left_line = Line()
    right_line = Line()

    # Lane detection
    if mode == 'video':
        # video
        processed_frame = 0

        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(filename).fl_image(process_pipeline)
        clip.write_videofile('project_video_output.mp4', audio=False)

    elif mode == 'image':
        img = mpimg.imread(filename)
        img_size = (img.shape[1], img.shape[0])

        # load calibration and top-down transform parameters
        mtx, dist, M, Minv = load_params('calibration_params.pkl', 'top_down_params.pkl')
        img_undist = cv2.undistort(img, mtx, dist, None, mtx)

        # generate binary image
        binary = preprocess(img_undist)
        binary_TD = cv2.warpPerspective(binary, M, img_size)

        # fit polynomial lines
        fit_result, left_line, right_line = poly_fit(binary_TD, verbose=True)
        fit_result, left_line, right_line = fast_poly_fit(binary_TD, left_line, right_line, verbose=True)

        newwarp = remap(binary_TD, left_line.current_fit, right_line.current_fit, Minv)
        result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)

        # Measure curvature radius
        left_curverad = left_line.radius_of_curvature
        right_curverad = right_line.radius_of_curvature

        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        plt.figure('newwarp')
        plt.imshow(result)
        plt.show()

    else:
        print("Mode selected error...")