import numpy as np 
import PIL
import cv2 

from glob import glob

from .hog_features import get_hog_features 
from .sift_features import get_sift_features
from .get_color_histograms import get_color_features
from .utils import read_frame, annotations_for_frame

def get_pos_and_neg(df, max_car_size=0.1, neg_img_per_frame=10, max_size=40*40, W=1280, H=720, add_cars_train=True, add_cars_test=True, add_other_cars=True):
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
        cnt = 0
        split_list = [120, 280, 430, 580]
        while cnt < neg_img_per_frame:
            y = int(np.random.choice(np.array(range(120, 580))))
            if y >= split_list[0] and y <= split_list[1]:
                dy = 64
                dx = 64
                x = int(np.random.choice(W-dx))
            elif y >= split_list[1] and y <= split_list[2]:
                dy = 96
                dx = 96
                x = int(np.random.choice(W-dx))
            elif y >= split_list[2] and y <= split_list[3]:
                dy = 128
                dx = 128
                x = int(np.random.choice(W-dx))
            bin_patch = bin_img[y:y+dy, x:x+dx]
            if np.mean(bin_patch) <= max_car_size:
                new_img = img[y:y+dy, x:x+dx,:] 
                train_neg_img.append(new_img)
                cnt += 1
    if add_cars_train:
        other_car_paths = glob("cars_train/cars_train"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            try:
                train_pos_img.append(new_img)
            except:
                continue
    if add_cars_test: 
        other_car_paths = glob("cars_test/cars_test"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            try:
                train_pos_img.append(new_img)
            except:
                continue
    if add_other_cars: 
        other_car_paths = glob("other_vehicles/Far"+"/*") + glob("other_vehicles/Left"+"/*") + glob("other_vehicles/MiddleClose"+"/*") + glob("other_vehicles/Right"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            try:
                train_pos_img.append(new_img)
            except:
                continue
        other_non_car_paths = glob("other_non_vehicles/Far"+"/*") + glob("other_non_vehicles/Left"+"/*") + glob("other_non_vehicles/MiddleClose"+"/*") + glob("other_non_vehicles/Right"+"/*")
        for non_car_path in other_non_car_paths:    
            img = np.asarray(PIL.Image.open(non_car_path))
            try:
                train_neg_img.append(new_img)
            except:
                continue
    return train_pos_img, train_neg_img

def get_features(train_pos_img, train_neg_img, hog_desc, sift_tools, winSize, use_hog=True, use_sift=False, use_spatial=True, use_color=True):
    train_pos_features = [0]*len(train_pos_img)
    train_neg_features = [0]*len(train_neg_img)
    for i in range(len(train_pos_img)):
        new_img = train_pos_img[i]  
        train_features_list = []
        #order : spatial, color, hog, sift        
        if use_spatial:
            spatial_features = cv2.resize(new_img, (16, 16)).flatten()
            train_features_list.append(spatial_features)        
        if use_color:
            color_features = get_color_features(new_img, winSize)
            train_features_list.append(color_features)
        if use_hog:
            hog_features = get_hog_features(hog_desc, new_img, winSize)
            train_features_list.append(hog_features)
        if use_sift and len(sift_tools) == 3:
            sift, vocab, vocab_size = sift_tools[0], sift_tools[1], sift_tools[2]
            sift_features = get_sift_features(sift, vocab, vocab_size, new_img)
            train_features_list.append(sift_features)
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
        if use_sift and len(sift_tools) == 3:
            sift, vocab, vocab_size = sift_tools[0], sift_tools[1], sift_tools[2]
            sift_features = get_sift_features(sift, vocab, vocab_size, new_img)
            train_features_list.append(sift_features)
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