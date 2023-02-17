import pandas as pd

import os

from utils import show_annotation, run_length_encoding, bounding_boxes_to_mask
from ipywidgets import interact, widgets


df_ground_truth = pd.read_csv('./train.csv')
df_ground_truth.head()

H, W = 720, 1280
N = len(df_ground_truth)

def f_display(frame_id):
    show_annotation(df_ground_truth, frame_id)
    
interact(f_display, frame_id=widgets.IntSlider(min=0, max=N-1, step=1, value=0))

bounding_boxes = [[0, 0, 5, 5], [5, 5, 5, 5]]

test_files = sorted(os.listdir('./test/'))

rows = []

for file_name in test_files:

    rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes, H, W))
    rows.append(['test/' + file_name, rle])

df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
df_prediction.to_csv('sample_submission.csv')