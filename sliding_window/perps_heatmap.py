import cv2 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler

class Heatmap():
    
    def __init__(self,original_image,threshold=0.63):
        
        # Mask attribute is the heatmap initialized with zeros
        self.mask = np.zeros(original_image.shape[:2])
        self.threshold = threshold
    
    # Increase value of region function will add some heat to heatmap
    def incValOfReg(self, coords):
        w1,w2,h1,h2 = coords
        self.mask[h1:h2,w1:w2] = self.mask[h1:h2,w1:w2] + 1
    
    def compileHeatmap(self, plot=True):
        
        # As you know,pixel values must be between 0 and 255 (uint8)
        # Now we'll scale our values between 0 and 255 and convert it to uint8
        
        # Scaling between 0 and 1 
        self.mask[0:120, :] = np.min(self.mask)
        self.mask[580:720, :] = np.min(self.mask) 
        self.mask = np.clip(self.mask, 0, 255)
        
        if plot:
            plt.matshow(self.mask)
            plt.title("mask with threshold="+str(self.threshold))
            plt.show()     
        
        self.mask = cv2.inRange(self.mask, self.threshold, 255) 
        mask_std = self.mask.std(ddof=1)
        if mask_std != 0.0:
            self.mask = (self.mask-self.mask.mean())/mask_std
        if plot:
            plt.matshow(self.mask)
            plt.title("mask with threshold="+str(self.threshold))
            plt.show()     
        self.mask = cv2.inRange(self.mask, np.max([self.mask.std(), 1]), np.max(self.mask))
        
        return self.mask