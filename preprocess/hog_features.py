import cv2

def return_hog_descriptor(winSize=(64, 64)):
    winSize = winSize
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hog_desc = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog_desc

def get_hog_features(hog_desc, img, winSize=(64,64)):
    res_img = cv2.cvtColor(cv2.resize(img, winSize),cv2.COLOR_BGR2GRAY)
    features = hog_desc.compute(res_img)
    return features


def get_hog_img(hog_desc, train_pos_img, train_neg_img, winSize=(64,64)):
    train_pos_hog = [0]*len(train_pos_img)
    train_neg_hog = [0]*len(train_neg_img)
    for i in range(len(train_pos_img)):
        new_img = train_pos_img[i]  
        new_img = cv2.cvtColor(cv2.resize(new_img, winSize),cv2.COLOR_BGR2GRAY)
        new_img = hog_desc.compute(new_img)   
        train_pos_hog[i] = new_img 
    for i in range(len(train_neg_img)):
        new_img = train_neg_img[i]  
        new_img = cv2.cvtColor(cv2.resize(new_img, winSize),cv2.COLOR_BGR2GRAY)
        new_img = hog_desc.compute(new_img)
        train_neg_hog[i] = new_img 
    return train_pos_hog, train_neg_hog
