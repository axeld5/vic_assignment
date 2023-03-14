import cv2
import numpy as np

from .get_color_histograms import get_color_features

def slideExtract(img, rescale_param, hog_desc, use_hog=True, use_spatial=True, use_color=True):
    
    # We'll store coords and features in these lists
    coords = []
    features = []

    winSize = (64, 64)
    x_step = int(16*rescale_param)
    y_step = int(16*rescale_param)

    h_org, w_org = img.shape[:2]
    h_new, w_new = int(h_org*rescale_param), int(w_org*rescale_param)

    image = cv2.resize(img, (w_new, h_new))

    hIm,wIm = image.shape[:2] 
    for w1,w2 in zip(range(0,wIm-winSize[0],x_step),range(winSize[0],wIm,x_step)):       
        for h1,h2 in zip(range(0, hIm-winSize[1],y_step),range(winSize[1], hIm, y_step)):
            window = image[h1:h2,w1:w2,:]   
            features_list = []         
            if use_spatial:
                spatial_window = cv2.resize(window, (16, 16)).flatten()         
                features_list.append(spatial_window)  
            window = cv2.resize(window, (64, 64))
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
