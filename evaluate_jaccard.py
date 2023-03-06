import time

from sklearn.metrics import jaccard_score

from preprocess.data_info import get_bin_masks, get_bin_mask_from_bbox_list
from sliding_window.detect import detect 

def evaluate_jaccard(img_list, train_bin_masks, hog_desc, sift_tools, use_hog, use_spatial, use_sift, use_color, 
                    clf, winSize, neg_max_proba, pos_max_proba, step):
    jaccard_scores = [0]*len(img_list)
    start = time.time()
    for i, img in enumerate(img_list):
        detected, bounding_boxes = detect(img, hog_desc, sift_tools, use_hog=use_hog, use_spatial=use_spatial, use_sift=use_sift, use_color=use_color,
                        clf=clf, winSize=winSize, neg_max_proba=neg_max_proba, pos_max_proba=pos_max_proba, step=step)
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