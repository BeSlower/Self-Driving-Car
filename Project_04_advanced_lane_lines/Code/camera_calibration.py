import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

calibration_params = 'calibration_params.pkl'
path = os.path.join(os.getcwd(), calibration_params)

import glob
images = glob.glob('../camera_cal/calibration1.jpg')

if not os.path.exists(path):
    nrows = 6
    ncols = 9
    data = {}

    # arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []
    objp = np.zeros((nrows*ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ncols, nrows), None)
        # if corners are found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (ncols, nrows), corners, ret)
            #plt.imshow(img)

    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    data['mtx'] = mtx
    data['dist'] = dist
    data['rvecs'] = rvecs
    data['tvecs'] = tvecs

    # store calibration parameters
    with open(calibration_params, 'wb') as output:
        pickle.dump(data, output)
else:
    with open(calibration_params, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    mtx = data['mtx']
    dist = data['dist']
    # show undistorted images
    for fname in images:
        img = mpimg.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        plt.imshow(dst)
        plt.title('undistorted image')
        plt.show()