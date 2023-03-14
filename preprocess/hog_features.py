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
