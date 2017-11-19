from utils import *

params_config = {
    'color_space': 'YCrCb',      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 9,                 # HOG orientations
    'pix_per_cell': 8,           # HOG pixels per cell
    'cell_per_block': 2,         # HOG cells per block
    'hog_channel': 'ALL',            # Can be 0, 1, 2, or "ALL"
    'spatial_size': (32, 32),    # Spatial binning dimensions
    'hist_bins': 16,             # Number of histogram bins
    'spatial_feat': True,        # Spatial features on or off
    'hist_feat': True,           # Histogram features on or off
    'hog_feat': True,            # HOG features on or off
}

image_size = [720, 1280]

window_list = []
x_start_stop = [image_size[1] * 2 // 5 + 90, [1] * 9 // 10]
y_start_stop = [image_size[0] * 1 // 2 + 45, image_size[0] * 3 // 5] 
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                   xy_window=(80, 50), xy_overlap=(0.8, 0.8), window_list = window_list)

x_start_stop = [image_size[1] * 2 // 5 + 90, image_size[1] * 9 // 10] 
y_start_stop = [image_size[0] * 1 // 2 + 45, image_size[0] * 3 // 5] 
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                   xy_window=(90, 60), xy_overlap=(0.8, 0.8), window_list = window_list)

x_start_stop = [image_size[1] * 3 // 5, None] 
y_start_stop = [image_size[0] * 1 // 2 + 45, image_size[0] * 7 // 10]
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                   xy_window=(100, 50), xy_overlap=(0.8, 0.8), window_list = window_list)

x_start_stop = [image_size[1] * 2 // 3, None]
y_start_stop = [image_size[0] * 2 // 3, None]
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                   xy_window=(120, 60), xy_overlap=(0.8, 0.8), window_list = window_list)

x_start_stop = [image_size[1] * 2 // 3, None]
y_start_stop = [image_size[0] * 2 // 3, None]
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=(150, 125), xy_overlap=(0.6, 0.6), window_list = window_list)

x_start_stop = [image_size[1] * 2 // 3, None]
y_start_stop = [image_size[0] * 2 // 3, None]
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=(200, 150), xy_overlap=(0.6, 0.6), window_list = window_list)

x_start_stop = [image_size[1] * 4 // 10, image_size[1] * 2 // 3] # Min and max in x to search in slide_window()
y_start_stop = [image_size[0] * 2 // 3, image_size[0] * 8 // 10] # Min and max in y to search in slide_window()
window_list = slide_window(image, x_start_stop = x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=(250, 135), xy_overlap=(0.5, 0.5), window_list = window_list)

