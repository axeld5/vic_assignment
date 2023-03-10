import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.persp_data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list, get_features
from preprocess.hog_features import return_hog_descriptor
from sliding_window.rescale_detect import detect 
from evaluate_jaccard import evaluate_jaccard


if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)
    max_size=40*40

    #get images block
    start = time.time()
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth, max_car_size=0, neg_img_per_frame=6, max_size=max_size, add_other_cars=True, add_other_non_cars=True, flip=False)
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
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
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
    #clf = HistGradientBoostingClassifier()
    clf = XGBClassifier()
    #6, 0.07, 500, 0.7
    params = { 'max_depth': [6,10],
           'learning_rate': [0.01, 0.07, 0.3],
           'n_estimators': [100, 500],
           'colsample_bytree': [0.3, 0.7]}
    skf = StratifiedKFold(n_splits=3)
    clf = GridSearchCV(estimator=clf, 
                    param_grid=params,
                    scoring='accuracy', 
                    cv=skf,
                    verbose=1)
    clf.fit(x_train, y_train)
    print("Best parameters:", clf.best_params_)
    #clf = LinearSVC()
    #clf = SVC()
    #clf = SVC(probability=True)
    print(time.time() - start)
    start = time.time()
    y_pred = clf.predict(x_test)
    print(time.time() - start)
    print("Accuracy score of model is ",f1_score(y_pred=y_pred,y_true=y_test)*100)