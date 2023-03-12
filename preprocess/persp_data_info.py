import numpy as np 
import PIL
import matplotlib.pyplot as plt
import cv2 

from glob import glob

from .hog_features import get_hog_features 
from .get_color_histograms import get_color_features
from .sift_features import get_sift_features
from .canny_binning import get_canny_features
from .utils import read_frame, annotations_for_frame


def get_pos_and_neg(df, max_car_size=0.1, neg_img_per_frame=10, max_size=40*40, W=1280, H=720, add_other_cars=True, add_other_non_cars=True, flip=True):
    train_pos_img = []
    train_neg_img = []
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        bin_img = np.zeros((H,W))
        for x, y, dx, dy in bbs:
            bin_img[y:y+dy, x:x+dx] = 1
            if dy*dx < max_size:
                continue
            new_img = img[y:y+dy, x:x+dx,:]
            train_pos_img.append(new_img)
            if flip:
                flipped_img = cv2.flip(new_img, 3)
                train_pos_img.append(flipped_img)
        cnt = 0
        while cnt < neg_img_per_frame:
            dy = np.random.choice([128, 192, 256])
            dx = dy 
            x = int(np.random.choice(W-dx))
            y = int(np.random.choice(H-dy))
            bin_patch = bin_img[y:y+dy, x:x+dx]
            if np.mean(bin_patch) <= max_car_size:
                new_img = img[y:y+dy, x:x+dx,:] 
                train_neg_img.append(new_img)
                if flip:
                    flipped_img = cv2.flip(new_img, 3)
                    train_neg_img.append(flipped_img)
                cnt += 1
    if add_other_cars: 
        other_car_paths = glob("vehicles/GTI_Far"+"/*") + glob("vehicles/GTI_Right"+"/*") + glob("vehicles/GTI_Left"+"/*") + glob("vehicles/GTI_MiddleClose"+"/*") + glob("vehicles/KITTI_extracted"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            train_pos_img.append(img)
            if flip:
                flipped_img = cv2.flip(img, 3)
                train_pos_img.append(flipped_img)
    if add_other_non_cars:
        other_non_car_paths = glob("non_vehicles/GTI"+"/*") + glob("non_vehicles/Extras"+"/*")
        for non_car_path in other_non_car_paths:    
            img = np.asarray(PIL.Image.open(non_car_path))
            train_neg_img.append(img)
            if flip:
                flipped_img = cv2.flip(img, 3)
                train_neg_img.append(flipped_img)
    return train_pos_img, train_neg_img

def get_features(train_pos_img, train_neg_img, hog_desc, winSize, use_hog=True, use_spatial=True, use_color=True, use_sift=True, sift_tools=[], use_canny=True):
    train_pos_features = [0]*len(train_pos_img)
    train_neg_features = [0]*len(train_neg_img)
    for i in range(len(train_pos_img)):
        new_img = train_pos_img[i]  
        train_features_list = []
        #order : spatial, color, hog      
        if use_spatial:
            spatial_features = cv2.resize(new_img, (16, 16)).flatten()
            train_features_list.append(spatial_features)        
        if use_color:
            color_features = get_color_features(new_img, winSize)
            train_features_list.append(color_features)
        if use_hog:
            hog_features = get_hog_features(hog_desc, new_img, winSize)
            train_features_list.append(hog_features)
        if use_sift:
            sift, vocab, vocab_size = sift_tools[0], sift_tools[1], sift_tools[2]
            sift_features = get_sift_features(sift, vocab, vocab_size, new_img)
            train_features_list.append(sift_features)
        if use_canny:
            canny_features = get_canny_features(new_img)
            train_features_list.append(canny_features)
        train_pos_features[i] = np.concatenate(train_features_list)
    for i in range(len(train_neg_img)):
        new_img = train_neg_img[i]  
        train_features_list = []
        if use_spatial:
            spatial_features = cv2.resize(new_img, (16, 16)).flatten()
            train_features_list.append(spatial_features)        
        if use_color:
            color_features = get_color_features(new_img, winSize)
            train_features_list.append(color_features)
        if use_hog:
            hog_features = get_hog_features(hog_desc, new_img, winSize)
            train_features_list.append(hog_features)
        if use_sift:
            sift, vocab, vocab_size = sift_tools[0], sift_tools[1], sift_tools[2]
            sift_features = get_sift_features(sift, vocab, vocab_size, new_img)
            train_features_list.append(sift_features)
        if use_canny:
            canny_features = get_canny_features(new_img)
            train_features_list.append(canny_features)
        train_neg_features[i] = np.concatenate(train_features_list)
    return train_pos_features, train_neg_features

def extract_bboxes(df):
    bboxes = []
    for frame in range(len(df.values.tolist())):
        bbs = annotations_for_frame(df, frame)
        bboxes.append(bbs)
    return bboxes

def get_bin_masks(df, W=1280, H=720):
    bin_img_list = []
    for frame in range(len(df.values.tolist())):
        bbs = annotations_for_frame(df, frame)
        bin_img = np.zeros((H,W))
        bbs = annotations_for_frame(df, frame)
        for x, y, dx, dy in bbs:
                bin_img[y:y+dy, x:x+dx] = 1
        bin_img_list.append(bin_img)
    return bin_img_list

def get_bin_mask_from_bbox_list(bbox_list, W=1280, H=720): 
    bin_img = np.zeros((H,W))
    for x, y, dx, dy in bbox_list: 
        bin_img[y:y+dy, x:x+dx] = 1 
    return bin_img