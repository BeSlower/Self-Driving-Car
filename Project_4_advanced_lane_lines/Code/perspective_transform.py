import cv2
import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

img = mpimg.imread('../test_images/straight_lines1.jpg')

# --------------------------
#   Distortion Correction
# --------------------------
with open('calibration_params.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)
mtx = data['mtx']
dist = data['dist']

img_undist = cv2.undistort(img, mtx, dist, None, mtx)

# --------------------------
#   Perspective Transform
# --------------------------
offset = 100
img_size = (img.shape[1], img.shape[0])

# for source points, I grabbed four points from test image
raw_points = np.float32([[583, 455], [699, 455], [193, 720], [1113, 720]])
# destination points
dst = np.float32([[200, 0],
                  [1000, 0],
                  [200, 720],
                 [1000, 720]])
M = cv2.getPerspectiveTransform(raw_points, dst)
Minv = cv2.getPerspectiveTransform(dst, raw_points)

# warp image into top-down view
top_down = cv2.warpPerspective(img_undist, M, img_size)

# plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=10)
plt.show()

# store perspective parameters
data = {'M': M,
        'Minv': Minv}

with open('top_down_params.pkl', 'wb') as output:
    pickle.dump(data, output)