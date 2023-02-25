import numpy as np 

from .utils import read_frame, annotations_for_frame

def get_pos_and_neg(df, neg_img_per_frame=8, max_car_size=0.1, max_size=40*40, dx_neg_base=20, dy_neg_base=20, W=1280, H=720):
    train_pos_img = []
    train_neg_img = []
    for frame in range(len(df.values.tolist())):
        img = np.asarray(read_frame(df, frame))
        bbs = annotations_for_frame(df, frame)
        bin_img = np.zeros_like(img)
        for x, y, dx, dy in bbs:
            bin_img[y:y+dy, x:x+dx,:] = 1
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
            bin_patch = bin_img[y:y+dy, x:x+dx,:]
            if np.mean(bin_patch) < max_car_size:
                new_img = img[y:y+dy, x:x+dx,:] 
                train_neg_img.append(new_img)
                cnt += 1
    return train_pos_img, train_neg_img