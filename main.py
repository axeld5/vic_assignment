import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score
from xgboost import XGBClassifier

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list, get_features
from preprocess.hog_features import return_hog_descriptor
from sliding_window.detect import detect 
from evaluate_jaccard import evaluate_jaccard


if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)
    max_size=30*30

    #get images block
    start = time.time()
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth, max_car_size=0, neg_img_per_frame=8, max_size=max_size, add_other_cars=True, add_other_non_cars=True)
    print(time.time() - start)
    print("imgs secured")


    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)

    use_hog = True 
    use_spatial = True
    use_color = True

    #apply hog block
    start = time.time()
    train_pos_features, train_neg_features = get_features(train_pos_img, train_neg_img, hog_desc, winSize=winSize,
                                                          use_hog=use_hog, use_spatial=use_spatial, use_color=use_color)
    print(time.time() - start)
    #get dataset block
    train_pos_labels = np.ones(len(train_pos_img))
    print(len(train_pos_labels))
    train_neg_labels = np.zeros(len(train_neg_img))
    print(len(train_neg_labels))

    x = np.asarray(train_pos_features + train_neg_features)
    y = np.asarray(list(train_pos_labels) + list(train_neg_labels))

    print("Shape of image set",x.shape)
    print("Shape of labels",y.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #train model block
    
    start = time.time()
    clf = XGBClassifier(max_depth=6, learning_rate=0.07, n_estimators=500, colsample_bytree=0.7)
    clf.fit(x_train, y_train)
    clf.save_model('0001.model')
    print(time.time() - start)
    start = time.time()
    y_pred = clf.predict(x_test)
    print(time.time() - start)
    print("Accuracy score of model is ",f1_score(y_pred=y_pred,y_true=y_test)*100)


    #sliding window block

    values = df_ground_truth.values.tolist()
    idx = np.random.choice(len(values))
    image = np.asarray(PIL.Image.open(values[idx][0]))
    train_bin_masks = get_bin_masks(df_ground_truth)
    threshold = 5
    proba_thresh = 0.93
    rescale_params = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
    do_test = True
    if not do_test:
        start = time.time()
        plot = False
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
        
        
        testing = False
        if testing:
            test_files = sorted(os.listdir('test/'))
            for i in range(10):
                idx = np.random.choice(len(test_files))
                test_img = np.asarray(PIL.Image.open('test/'+test_files[idx]))
                detected, bounding_boxes = detect(test_img, rescale_params, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color, 
                                                clf=clf, threshold=threshold, proba_thresh=proba_thresh, max_size=max_size, plot=True)
                plt.imshow(detected)
                plt.show()

    get_pred = False
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