import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

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

def preprocess(img_undist):
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

    # plot
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(40, 40))
    # ax1.set_title('undistorted image')
    # ax1.imshow(img_undist)
    #
    # ax2.set_title('color thresholding')
    # ax2.imshow(s_channel_binary, cmap='gray')
    #
    # ax3.set_title('x gradient thresholding')
    # ax3.imshow(sxbinary, cmap='gray')
    #
    # ax4.set_title('combined binary')
    # ax4.imshow(combine_binary, cmap='gray')

    return combine_binary

def poly_fit(binary_TD):
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

    # generate x and y values for plotting
    ploty = np.linspace(0, binary_TD.shape[0]-1, binary_TD.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.figure('Sliding Windows Polyfit')
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    # fit result
    result = {'left_line': {'x': leftx, 'y': lefty, 'fit': left_fit},
            'right_line': {'x': rightx, 'y': righty, 'fit': right_fit}}

    return result

def fast_poly_fit(binary_TD, left_fit, right_fit):
    '''
        Assume you now have a new warped binary image
        from the next frame of video (also called "binary_TD")
        It's now much easier to find line pixels
    '''
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

    # fit result
    result = {'left_line': {'x': leftx, 'y': lefty, 'fit': left_fit},
            'right_line': {'x': rightx, 'y': righty, 'fit': right_fit}}

    return result

def curvature_estimate(y, x, y_eval):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    # Calculate the new radii of curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit_cr[0])

    return curverad

def main():

    mode = 'video'
    filename = 'project_video.mp4'

    if mode == 'video':
        # load calibration and top-down transform parameters
        mtx, dist, M, Minv = load_params('calibration_params.pkl', 'top_down_params.pkl')

        # video
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(filename)
        #img_size = (frame.shape[1], frame.shape[0])

        for frame_idx, img in enumerate(clip.iter_frames()):
            # Capture frame-by-frame
            if frame_idx == 0:
                img_size = (img.shape[1], img.shape[0])

            img_undist = cv2.undistort(img, mtx, dist, None, mtx)
            binary = preprocess(img_undist)
            binary_TD = cv2.warpPerspective(binary, M, img_size)

            # fit polynomial lines
            if True:#frame_idx == 0:
                fit_result = poly_fit(binary_TD)
            else:
                fit_result = fast_poly_fit(binary_TD, left_fit, right_fit)

            # Measure curvature radius
            left_fit = fit_result['left_line']['fit']
            right_fit = fit_result['right_line']['fit']

            left_curverad = curvature_estimate(fit_result['left_line']['y'], fit_result['left_line']['x'],
                                               binary_TD.shape[0])
            right_curverad = curvature_estimate(fit_result['left_line']['y'], fit_result['left_line']['x'],
                                                binary_TD.shape[0])

            print(left_curverad, 'm', right_curverad, 'm')
            plt.pause(0.05)

    elif mode == 'image':

        img = cv2.imread('../test_images/test3.jpg')
        img_size = (img.shape[1], img.shape[0])

        # load calibration and top-down transform parameters
        mtx, dist, M, Minv = load_params('calibration_params.pkl', 'top_down_params.pkl')
        img_undist = cv2.undistort(img, mtx, dist, None, mtx)

        # generate binary image
        binary = preprocess(img_undist)
        binary_TD = cv2.warpPerspective(binary, M, img_size)

        # fit polynomial lines
        fit_result = poly_fit(binary_TD)

        # Measure curvature radius
        left_curverad = curvature_estimate(fit_result['left_line']['y'], fit_result['left_line']['x'], binary_TD.shape[0])
        right_curverad = curvature_estimate(fit_result['left_line']['y'], fit_result['left_line']['x'], binary_TD.shape[0])

        # fast fit a polynomial line for the next frame
        # left_fit, right_fit, left_curverad, right_curverad = fast_poly_fit(binary_TD, left_fit, right_fit)

        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')

        plt.figure()
        plt.title('binary top-down')
        plt.imshow(binary_TD, cmap='gray')
        plt.show()

    else:
        print("Mode selected error...")

if __name__ == '__main__':
    main()