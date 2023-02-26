import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd  

from preprocess.utils import read_frame, annotations_for_frame

def get_width_histo(df, W=1280, H=720):
    bin_img = np.zeros((H, W)) 
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        for x, y, dx, dy in bbs:
            bin_img[y:y+dy, x:x+dx] += 1
    width_img = np.mean(bin_img, axis=0)
    x = np.arange(W) 
    plt.plot(x, width_img)
    plt.show()


def get_height_histo(df, W=1280, H=720):
    bin_img = np.zeros((H, W)) 
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        for x, y, dx, dy in bbs:
            bin_img[y:y+dy, x:x+dx] += 1
    height_img = np.mean(bin_img, axis=1)
    x = np.arange(H) 
    plt.plot(x, height_img)
    plt.show()

if __name__ == "__main__":
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    get_width_histo(df_ground_truth)
    get_height_histo(df_ground_truth)