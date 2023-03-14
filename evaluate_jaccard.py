import time

from sklearn.metrics import jaccard_score

from preprocess.data_info import get_bin_masks, get_bin_mask_from_bbox_list
from sliding_window.detect import detect 

def evaluate_jaccard(img_list, train_bin_masks, rescale_params, scaler, hog_desc, use_hog, use_spatial, use_color,
                        clf, threshold, max_size=30*30, plot=False):
    jaccard_scores = [0]*len(img_list)
    start = time.time()
    for i, img in enumerate(img_list):
        detected, bounding_boxes = detect(img, rescale_params=rescale_params, scaler=scaler, hog_desc=hog_desc, use_hog=use_hog, use_spatial=use_spatial, use_color=use_color,
                        clf=clf, threshold=threshold, max_size=max_size, plot=False)
        train_bin_mask_img = train_bin_masks[i]
        pred_bin_mask_img = get_bin_mask_from_bbox_list(bounding_boxes)
        if len(train_bin_mask_img) == 0:
            if len(pred_bin_mask_img) == 0:
                jaccard_score[i] = 1 
            else:
                jaccard_score[i] = 0
        else:
            jaccard_scores[i] = jaccard_score(train_bin_mask_img, pred_bin_mask_img, average="micro")
    print(time.time() - start)
    return sum(jaccard_scores)/len(jaccard_scores)