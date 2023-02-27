import numpy as np
import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt 
import seaborn as sns
import PIL
import cv2
import pickle

from glob import glob
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, jaccard_score

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list
from preprocess.hog import return_hog_descriptor, get_hog_img
from sliding_window.detect import detect 

#params we can play on: all params of get_pos_and_neg, neg_max_proba and pos_max_proba

if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)

    #get images block
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth, max_car_size=0, neg_img_per_frame=5, method="random", dx_neg_base=30, dy_neg_base=20, max_size=64*64, add_cars_test=False, add_cars_train=False)

    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)

    #apply hog block
    train_pos_hog, train_neg_hog = get_hog_img(hog_desc, train_pos_img, train_neg_img, winSize)

    #get dataset block
    train_pos_labels = np.ones(len(train_pos_img))
    print(len(train_pos_labels))
    train_neg_labels = np.zeros(len(train_neg_img))
    print(len(train_neg_labels))

    x = np.asarray(train_pos_hog + train_neg_hog)
    y = np.asarray(list(train_pos_labels) + list(train_neg_labels))

    print("Shape of image set",x.shape)
    print("Shape of labels",y.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #train model block

    #clf = HistGradientBoostingClassifier().fit(x_train, y_train)
    start = time.time()
    clf = SVC(probability=True).fit(x_train, y_train)
    # We'll use Cross Validation Grid Search to find best parameters.
    # Classifier will be trained using each parameter 
    clf.fit(x_train,y_train)
    print(time.time() - start)
    y_pred = clf.predict(x_test)
    print("Accuracy score of model is ",f1_score(y_pred=y_pred,y_true=y_test)*100)


    #sliding window block

    values = df_ground_truth.values.tolist()
    image = np.asarray(PIL.Image.open(values[0][0]))
    train_bin_mask = get_bin_masks(df_ground_truth)[0] 


    start = time.time()
    detected, bounding_boxes = detect(image, hog_desc, clf, winSize, neg_max_proba=0.9, pos_max_proba=0.1, step=25)
    print(time.time() - start)
    pred_bin_mask = get_bin_mask_from_bbox_list(bounding_boxes)
    print(jaccard_score(train_bin_mask, pred_bin_mask, average="micro"))
    plt.imshow(detected)
    plt.show()

    test_files = sorted(os.listdir('test/'))
    test_img = np.asarray(PIL.Image.open('test/'+test_files[0]))
    detected, bounding_boxes = detect(test_img, hog_desc, clf, winSize, neg_max_proba=0.9, pos_max_proba=0.1, step=25)
    plt.imshow(detected)
    plt.show()

    """get_pred = True
    if get_pred:
        test_files = sorted(os.listdir('test/'))
        print(len(test_files))
        rows = []

        for i, file_name in enumerate(test_files):
            image = np.asarray(PIL.Image.open('test/'+file_name))
            _, bounding_boxes = detect(image, hog_desc, clf, winSize, neg_max_proba=0.6, pos_max_proba=0.4, step=25)
            rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes, H, W))
            rows.append(['test/' + file_name, rle])
            if i%10 == 0:
                print(i)
        df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
        df_prediction.to_csv('predicted_cars.csv')"""