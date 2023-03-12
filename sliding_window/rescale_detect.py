import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import time

from .rescale_extract import slideExtract 
from .rescale_heatmap import Heatmap

def detect(image, rescale_params, scaler, hog_desc, use_hog, use_color, use_spatial, use_sift, sift_tools, use_canny, 
           clf, threshold=4, proba_thresh=0.9, max_size=30*30, plot=True):
    
    # Extracting features and initalizing heatmap
    mask_list = []
    hIm, wIm = image.shape[:2]
    for j, param in enumerate(rescale_params):
        coords,features = slideExtract(image, rescale_param=param, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color, 
                                       use_sift=use_sift, sift_tools=sift_tools, use_canny=use_canny)
        features = scaler.transform(features)
        scaled_im = cv2.resize(image, (int(wIm*param), int(hIm*param)))
        htmp = Heatmap(scaled_im)
        decisions = clf.predict_proba(features)
        for i in range(len(features)):
            # If region is positive then add some heat
            if decisions[i][1] > proba_thresh:
                htmp.incValOfReg(coords[i])
        mask_list.append(cv2.resize(htmp.mask, (wIm, hIm)))
    # Compiling heatmap
    mask = np.zeros((hIm, wIm))
    for mask_elem in mask_list:
        mask += mask_elem
    mask = compute_mask(mask, threshold, proba_thresh, plot)
    cont,_ = cv2.findContours(mask,1,2)[:2]
    bounding_boxes = []
    for c in cont:
        # If a contour is small don't consider it
        if cv2.contourArea(c) < max_size:
            continue
        (x_c,y_c,w,h) = cv2.boundingRect(c)
        bounding_boxes.append([x_c, y_c, w, h])
        image = cv2.rectangle(image,(x_c,y_c),(x_c+w,y_c+h),(255),2)
    
    return image, bounding_boxes

def compute_mask(mask, threshold, proba_thresh, plot):
    mask = np.clip(mask, 0, 255)
    mask[0:120, :] = np.min(mask)
    mask[580:720, :] = np.min(mask)
    if plot:
        plt.matshow(mask)
        plt.title("mask with threshold="+str(threshold)+", proba_thresh="+str(proba_thresh))
        plt.show()    
    mask = cv2.inRange(mask, threshold, 255) 
    mask_std = mask.std(ddof=1)
    if plot:
        plt.matshow(mask)
        plt.title("thresholded mask with threshold="+str(threshold)+", proba_thresh="+str(proba_thresh))
        plt.show()  
    """if mask_std != 0.0:
        mask = (mask-mask.mean())/mask_std     
    try: 
        mask = cv2.inRange(mask, np.max([mask.std(), 1]), np.max(mask))
    except:
        mask = np.zeros_like(mask)"""
    return mask