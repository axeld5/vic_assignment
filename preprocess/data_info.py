import numpy as np 
import PIL

from glob import glob

from .utils import read_frame, annotations_for_frame

def get_pos_and_neg(df, max_car_size=0.1, neg_img_per_frame=10, dx_neg_base=15, dy_neg_base=15, max_size=40*40, W=1280, H=720, add_cars_train=True, add_cars_test=True):
    train_pos_img = []
    train_neg_img = []
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        bin_img = np.zeros((H,W))
        for x, y, dx, dy in bbs:
            bin_img[y:y+dy, x:x+dx] = 1
            if dy*dx < max_size:
                continue
            new_img = img[y:y+dy, x:x+dx,:]
            train_pos_img.append(new_img)
        cnt = 0
        while cnt < neg_img_per_frame:
            dy = dy_neg_base*(cnt+1)
            dx = dx_neg_base*(cnt+1)
            x = int(np.random.choice(W-dx))
            y = int(np.random.choice(H-dy))
            bin_patch = bin_img[y:y+dy, x:x+dx]
            if np.mean(bin_patch) < max_car_size:
                new_img = img[y:y+dy, x:x+dx,:] 
                train_neg_img.append(new_img)
                cnt += 1
    if add_cars_train:
        other_car_paths = glob("cars_train/cars_train"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            try:
                train_pos_img.append(new_img)
            except:
                continue
    if add_cars_test: 
        other_car_paths = glob("cars_test/cars_test"+"/*")
        for car_path in other_car_paths:    
            img = np.asarray(PIL.Image.open(car_path))
            try:
                train_pos_img.append(new_img)
            except:
                continue
    return train_pos_img, train_neg_img

