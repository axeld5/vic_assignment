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

def get_img_info(df, W=1280, H=720):
    img_info = {"avg_width":[], "avg_height":[]}
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        for x, y, dx, dy in bbs:
            new_img = img[y:y+dy, x:x+dx, :]
            img_info["avg_height"].append(new_img.shape[0])
            img_info["avg_width"].append(new_img.shape[1])
    return np.mean(np.array(img_info["avg_height"])), np.argmax(np.array(img_info["avg_height"])), np.mean(np.array(img_info["avg_width"])), np.argmax(np.array(img_info["avg_width"]))



if __name__ == "__main__":
    df_ground_truth = pd.read_csv('train.csv')
    W = 1280
    H = 720
    print(get_img_info(df_ground_truth))