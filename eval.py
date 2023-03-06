import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 
import PIL
import cv2

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score

from preprocess.utils import run_length_encoding, bounding_boxes_to_mask
from preprocess.data_info import get_pos_and_neg, get_bin_masks, get_bin_mask_from_bbox_list, get_features
from preprocess.hog_features import return_hog_descriptor
from preprocess.sift_features import load_vocabulary
from sliding_window.detect import detect 
from evaluate_jaccard import evaluate_jaccard
from ensemble_classifier import EnsembleClassifier

#params we can play on: all params of get_pos_and_neg, neg_max_proba and pos_max_proba

if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)

    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)
    sift = cv2.SIFT_create()
    vocab_size = 250
    vocab = vocab = load_vocabulary("vocab.pkl")
    print("vocab done")

    sift_tools = [sift, vocab, vocab_size]
    use_hog = True 
    use_sift = True 
    use_spatial = True 
    use_color = True

    #train model block
    clf = EnsembleClassifier(use_svm=False, use_rf=True, use_xgb=True)
    clf.load_models()

    values = df_ground_truth.values.tolist()
    image = np.asarray(PIL.Image.open(values[0][0]))
    train_bin_masks = get_bin_masks(df_ground_truth)

    step = 25
    neg_max_proba = 0.9
    pos_max_proba = 0.1

    start = time.time()
    detected, bounding_boxes = detect(image, hog_desc, sift_tools, use_hog=use_hog, use_spatial=use_spatial, use_sift=use_sift, use_color=use_color,
                        clf=clf, winSize=winSize, neg_max_proba=neg_max_proba, pos_max_proba=pos_max_proba, step=step)
    print(time.time() - start)
    pred_bin_mask = get_bin_mask_from_bbox_list(bounding_boxes)
    print(jaccard_score(train_bin_masks[0], pred_bin_mask, average="micro"))
    plt.imshow(detected)
    plt.show()

    n_test = 100 
    img_list = [np.asarray(PIL.Image.open(values[i*5][0])) for i in range(n_test)]
    train_bin_mask = [train_bin_masks[i*5] for i in range(n_test)]
    print(evaluate_jaccard(img_list, train_bin_mask, hog_desc, sift_tools, use_hog=use_hog, use_spatial=use_spatial, use_sift=use_sift, use_color=use_color,
                        clf=clf, winSize=winSize, neg_max_proba=neg_max_proba, pos_max_proba=pos_max_proba, step=step))