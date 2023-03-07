import cv2 

from .perps_extract import slideExtract 
from .perps_heatmap import Heatmap

def detect(image, hog_desc, sift_tools, use_hog, use_sift, use_color, use_spatial, clf, winSize=(64, 64), pos_max_proba=0.9, step=30, max_size=40*40):
    
    # Extracting features and initalizing heatmap
    coords,features = slideExtract(image, hog_desc=hog_desc, sift_tools=sift_tools, winSize=winSize, step=step, 
                                   use_hog=use_hog, use_sift=use_sift, use_spatial=use_spatial, use_color=use_color)
    htmp = Heatmap(image)
    
    for i in range(len(features)):
        # If region is positive then add some heat
        try:
            proba = clf.predict_proba([features[i]])
            if proba[0][1] > pos_max_proba:
                htmp.incValOfReg(coords[i])
        except: 
            decision = clf.predict([features[i]])
            if decision[0] == 1:
                htmp.incValOfReg(coords[i])
                # Else remove some heat
            else:
                htmp.decValOfReg(coords[i])

    # Compiling heatmap
    mask = htmp.compileHeatmap()
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
