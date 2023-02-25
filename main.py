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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.data_info import get_pos_and_neg
from preprocess.hog import return_hog_descriptor, get_hog_img
from sliding_window.detect import detect 

if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)

    #get images block
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth)

    #get hog block
    hog_desc = return_hog_descriptor()

    #apply hog block
    train_pos_hog, train_neg_hog = get_hog_img(hog_desc, train_pos_img, train_neg_img)

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

    clf = HistGradientBoostingClassifier().fit(x_train, y_train)
    # We'll use Cross Validation Grid Search to find best parameters.
    # Classifier will be trained using each parameter 
    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)
    print("Accuracy score of model is ",f1_score(y_pred=y_pred,y_true=y_test)*100)


    #sliding window block

    values = df_ground_truth.values.tolist()
    image = np.asarray(PIL.Image.open(values[0][0]))


    start = time.time()
    detected, bounding_boxes = detect(image, hog_desc, clf)
    print(time.time() - start)
    plt.imshow(detected)
    plt.show()

    get_pred = False 
    if get_pred:
        test_files = sorted(os.listdir('test/'))
        print(len(test_files))
        rows = []

        for i, file_name in enumerate(test_files):
            image = np.asarray(PIL.Image.open('/content/drive/MyDrive/vic_class/assignment2/test/test/'+file_name))
            _, bounding_boxes = detect(image)
            rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes, H, W))
            rows.append(['test/' + file_name, rle])
            if i%10 == 0:
                print(i)