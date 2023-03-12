import cv2 

def get_canny_features(img, resize_window=(64,64)):
    resized_img = cv2.resize(img, resize_window)
    edges = cv2.Canny(resized_img, 100, 200)
    return edges.flatten()