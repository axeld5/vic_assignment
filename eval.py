import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2

from sklearn.metrics import jaccard_score
from xgboost import XGBClassifier

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.data_info import get_bin_masks, get_bin_mask_from_bbox_list
from preprocess.hog_features import return_hog_descriptor
from sliding_window.detect import detect 


if __name__ == "__main__":    
    df_ground_truth = pd.read_csv('train_copy.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)
    max_size=30*30

    start = time.time()
    clf = XGBClassifier(max_depth=6, learning_rate=0.07, n_estimators=500, colsample_bytree=0.7)
    clf.load_model('0004.model')

    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)

    scaler = None

    use_hog = True 
    use_spatial = True
    use_color = True

    #sliding window block

    values = df_ground_truth.values.tolist()
    idx = np.random.choice(len(values))
    image = np.asarray(PIL.Image.open(values[idx][0]))
    train_bin_masks = get_bin_masks(df_ground_truth)
    threshold_list = np.linspace(10, 50, 9, endpoint=True)
    threshold = 5
    proba_thresh = 0.93
    proba_thresh_list = np.linspace(0.6, 0.85, 6, endpoint=True)
    #rescale_params = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3]
    #rescale_params = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4]
    rescale_params = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    do_test = False
    if do_test:
        start = time.time()
        plot = True
        detected, bounding_boxes = detect(image, rescale_params, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                                        clf=clf, threshold=threshold, proba_thresh=proba_thresh, max_size=max_size, plot=plot)
        print(time.time() - start)
        pred_bin_mask = get_bin_mask_from_bbox_list(bounding_boxes)
        print("threshold="+str(threshold))
        print(jaccard_score(train_bin_masks[idx], pred_bin_mask, average="micro"))
        if plot:
            plt.imshow(detected)
            plt.title("threshold="+str(threshold))
            plt.show()
        
        
        testing = True
        if testing:
            test_files = sorted(os.listdir('test/'))
            for i in range(10):
                idx = np.random.choice(len(test_files))
                test_img = np.asarray(PIL.Image.open('test/'+test_files[idx]))
                detected, bounding_boxes = detect(test_img, rescale_params, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                                                clf=clf, threshold=threshold, proba_thresh=proba_thresh, max_size=max_size, plot=True)
                plt.imshow(detected)
                plt.show()

    get_pred = True
    if get_pred:
        test_files = sorted(os.listdir('test/'))
        print(len(test_files))
        rows = []

        for i, file_name in enumerate(test_files):
            image = np.asarray(PIL.Image.open('test/'+file_name))
            _, bounding_boxes = detect(image, rescale_params, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                                    clf=clf, threshold=threshold, proba_thresh=proba_thresh, max_size=max_size, plot=False)
            rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes, H, W))
            rows.append(['test/' + file_name, rle])
            if i%10 == 0:
                print(i)
        df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
        df_prediction.to_csv('predicted_cars.csv')