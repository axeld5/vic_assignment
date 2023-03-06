import cv2
import numpy as np

from .sift_features import get_sift_features
from .get_color_histograms import get_color_features

def slideExtract(image, hog_desc, sift_tools, winSize=(64, 64), step=30, use_hog=True, use_sift=False, use_spatial=True, use_color=True):
    
    # We'll store coords and features in these lists
    coords = []
    features = []
    
    hIm,wIm = image.shape[:2] 

    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1,w2 in zip(range(0,wIm-winSize[0],step),range(winSize[0],wIm,step)):       
        for h1,h2 in zip(range(0,hIm-winSize[1],step),range(winSize[1],hIm,step)):
            window = image[h1:h2,w1:w2,:]   
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
            if use_sift and len(sift_tools) == 3:
                sift, vocab, vocab_size = sift_tools[0], sift_tools[1], sift_tools[2]
                sift_features = get_sift_features(sift, vocab, vocab_size, window)
                features_list.append(sift_features)
            features_of_window = np.concatenate(features_list)
            coords.append((w1,w2,h1,h2))
            features.append(features_of_window)
    
    return (coords,np.asarray(features))
