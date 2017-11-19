import pickle
from code.utils import *
from code.config import params_config, window_list
from scipy.ndimage.measurements import label
import collections

def process_pipeline(image):

    draw_image = np.copy(image)
    draw_image2 = np.copy(image)
    heatmap = np.zeros_like(image[:, :, 0])

    image = image.astype(np.float32) / 255
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=params_config['y_start_stop'], xy_window=(96, 96), xy_overlap=(0.5, 0.5), window_list=window_list)

    hot_windows, scores = search_windows(image, windows, data['model'], data['x_scaler'], color_space=params_config['color_space'],
                                spatial_size=params_config['spatial_size'], hist_bins=params_config['hist_bins'],
                                orient=params_config['orient'], pix_per_cell=params_config['pix_per_cell'],
                                cell_per_block=params_config['cell_per_block'],
                                hog_channel=params_config['hog_channel'], spatial_feat=params_config['spatial_feat'],
                                hist_feat=params_config['hist_feat'], hog_feat=params_config['hog_feat'])

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    heatmap = add_heat(heatmap, hot_windows)
    heatmap_sequence.append(heatmap)
    heatmap_ave = sum(heatmap_sequence)
    heatmap_ave = apply_threshold(heatmap_ave, 8)

    labels = label(heatmap_ave)
    draw_img = draw_labeled_bboxes(draw_image2, labels)

    window_img = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
    
    return draw_img

if __name__ == '__main__':

    # load trained model and data scaled parameter
    with open('svm_model.pkl', 'rb') as f:
        data = pickle.load(f)

    # model selection
    model = 'video'
    filename = './project_video.mp4'

    # vehicle detection
    if model == 'video':
        # video
        heatmap_sequence = collections.deque(maxlen=10)

        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(filename).fl_image(process_pipeline)
        clip.write_videofile('project_video_output.mp4', audio=False)

    elif model == 'image':
        image = mpimg.imread('./test_images/test1.jpg')

        result = process_pipeline(image)

    else:
        print('Model selected error...')