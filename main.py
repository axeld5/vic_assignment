import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, jaccard_score

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.persp_data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list, get_features
from preprocess.hog_features import return_hog_descriptor
from sliding_window.perps_detect import detect 
from evaluate_jaccard import evaluate_jaccard

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


    #sliding window block

    values = df_ground_truth.values.tolist()
    image = np.asarray(PIL.Image.open(values[0][0]))
    train_bin_masks = get_bin_masks(df_ground_truth)
    threshold_list = np.linspace(5, 50, 10, endpoint=True)
    threshold = 4
    start = time.time()
    detected, bounding_boxes = detect(image, hog_desc,use_hog=use_hog, use_spatial=use_spatial,use_color=use_color,
                                clf=clf, threshold=threshold, max_size=30*30)
    print(time.time() - start)
    pred_bin_mask = get_bin_mask_from_bbox_list(bounding_boxes)
    print(jaccard_score(train_bin_masks[0], pred_bin_mask, average="micro"))
    plt.imshow(detected)
    plt.title("threshold="+str(threshold))
    plt.show()

    """n_test = 100 
    img_list = [np.asarray(PIL.Image.open(values[i*5][0])) for i in range(n_test)]
    train_bin_mask = [train_bin_masks[i*5] for i in range(n_test)]
    print(evaluate_jaccard(img_list, train_bin_mask, hog_desc, sift_tools, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                        clf=clf, threshold=threshold, step=step))"""

    test_files = sorted(os.listdir('test/'))
    test_img = np.asarray(PIL.Image.open('test/'+test_files[0]))
    detected, bounding_boxes = detect(test_img, hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                        clf=clf, threshold=threshold, max_size=30*30)
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