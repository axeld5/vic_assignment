import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.persp_data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list, get_features
from preprocess.hog_features import return_hog_descriptor
#params we can play on: all params of get_pos_and_neg, neg_max_proba and pos_max_proba

if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)

    #get images block
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth, max_car_size=0, neg_img_per_frame=12, max_size=30*30, add_cars_test=True, add_cars_train=True, add_other_cars=False)
    print("imgs secured")


    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)

    use_hog = True 
    use_spatial = True 
    use_color = True

    #apply hog block
    train_pos_features, train_neg_features = get_features(train_pos_img, train_neg_img, hog_desc, winSize=winSize,
                                                          use_hog=use_hog, use_spatial=use_spatial, use_color=use_color)

    #get dataset block
    train_pos_labels = np.ones(len(train_pos_img))
    print(len(train_pos_labels))
    train_neg_labels = np.zeros(len(train_neg_img))
    print(len(train_neg_labels))

    x = np.asarray(train_pos_features + train_neg_features)
    y = np.asarray(list(train_pos_labels) + list(train_neg_labels))

    print("Shape of image set",x.shape)
    print("Shape of labels",y.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #train model block
    
    start = time.time()
    clf = HistGradientBoostingClassifier().fit(x_train, y_train)
    #clf = SVC().fit(x_train, y_train)
    #clf = SVC(probability=True).fit(x_train, y_train)
    print(time.time() - start)
    y_pred = clf.predict(x_test)
    print("Accuracy score of model is ",f1_score(y_pred=y_pred,y_true=y_test)*100)