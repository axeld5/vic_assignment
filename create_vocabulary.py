import pandas as pd
import cv2 

from preprocess.data_info import get_pos_and_neg
from preprocess.hog_features import return_hog_descriptor
from preprocess.sift_features import save_vocabulary

#params we can play on: all params of get_pos_and_neg, neg_max_proba and pos_max_proba

if __name__ == "__main__":
    #starting block
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    N = len(df_ground_truth)

    #get images block
    train_pos_img, train_neg_img = get_pos_and_neg(df_ground_truth, max_car_size=0, neg_img_per_frame=12, max_size=50*50, add_cars_test=True, add_cars_train=True)

    winSize = (64, 64)
    #get hog block
    hog_desc = return_hog_descriptor(winSize)
    sift = cv2.SIFT_create()
    vocab_size = 250
    vocab = save_vocabulary(sift, train_pos_img + train_neg_img, vocab_size=vocab_size)
    print("vocab done")