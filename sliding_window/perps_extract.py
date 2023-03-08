import cv2
import numpy as np

from .get_color_histograms import get_color_features

def slideExtract(image, hog_desc, use_hog=True, use_spatial=True, use_color=True):
    
    # We'll store coords and features in these lists
    coords = []
    features = []
    
    hIm,wIm = image.shape[:2] 

    split_list = [120, 280, 430, 580]
    winSize_list = [(64, 64), (96, 96), (128, 128)]
    y_step_list = [2, 2, 2]
    x_step_list = [16, 24, 32]
    for i in range(3):
        winSize = winSize_list[i]
        y_step = y_step_list[i]
        x_step = x_step_list[i]
        for w1,w2 in zip(range(0,wIm-winSize[0],y_step),range(winSize[0],wIm,y_step)):       
            for h1,h2 in zip(range(split_list[i],split_list[i+1]-winSize[1],x_step),range(split_list[i]+winSize[1],split_list[i+1],x_step)):
                window = image[h1:h2,w1:w2,:]   
                window = cv2.resize(window, (64, 64))
                features_list = []         
                if use_spatial:
                    spatial_window = cv2.resize(window, (16, 16)).flatten()         
                    features_list.append(spatial_window)  
                if use_color: 
                    colors_of_window = get_color_features(window, winSize)
                    features_list.append(colors_of_window)
                window = cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
                if use_hog:
                    hog_of_window = hog_desc.compute(window)
                    features_list.append(hog_of_window)
                features_of_window = np.concatenate(features_list)
                coords.append((w1,w2,h1,h2))
                features.append(features_of_window)
    
    return (coords,np.asarray(features))
