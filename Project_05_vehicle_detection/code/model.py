import glob
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from config import params_config
import pickle

# Read in cars and notcars
cars = glob.glob('data/vehicles/*/*.png')
notcars = glob.glob('data/non-vehicles/*/*.png')


car_features = extract_features(cars, color_space=params_config['color_space'],
                                spatial_size=params_config['spatial_size'], hist_bins=params_config['hist_bins'],
                                orient=params_config['orient'], pix_per_cell=params_config['pix_per_cell'],
                                cell_per_block=params_config['cell_per_block'],
                                hog_channel=params_config['hog_channel'], spatial_feat=params_config['spatial_feat'],
                                hist_feat=params_config['hist_feat'], hog_feat=params_config['hog_feat'])

notcar_features = extract_features(notcars, color_space=params_config['color_space'],
                                spatial_size=params_config['spatial_size'], hist_bins=params_config['hist_bins'],
                                orient=params_config['orient'], pix_per_cell=params_config['pix_per_cell'],
                                cell_per_block=params_config['cell_per_block'],
                                hog_channel=params_config['hog_channel'], spatial_feat=params_config['spatial_feat'],
                                hist_feat=params_config['hist_feat'], hog_feat=params_config['hog_feat'])

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', params_config['orient'], 'orientations', params_config['pix_per_cell'],
      'pixels per cell and', params_config['cell_per_block'], 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

# model save
data = {'model': svc, 'x_scaler': X_scaler}

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(data, f)
