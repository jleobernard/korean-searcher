import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from com.leo.koreanparser.dl.utils.data_utils import read_image


def same_subs(expected_zone: np.ndarray, image_zone: np.ndarray) -> bool:
    threshold = 200
    expected_zone_thresholded = np.where(expected_zone > threshold, 1, 0)
    image_zone_thresholded = np.where(image_zone > threshold , 1, 0)
    ratio = (expected_zone_thresholded * image_zone_thresholded).sum() / (expected_zone_thresholded.sum() + 1e-6)
    return ratio > 0.75


annotation_file_name = "/opt/data/korean-subs/work/annotations_okay-not-okay.csv"
prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
directory = Path(annotation_file_name).resolve().parent
df_annotations_in = pd.read_csv(annotation_file_name)
data_out = []
curr_bb = None
first_image = None
first_bw_image = None
subs_frames_indices = []
subs_image = []
curr_subs_frame_indices = None
old_index = -1

def finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices):
    delta = 10
    max_height, max_width, _ = first_image.shape
    y0 = max(0, curr_bb[0] - delta)
    y1 = min(max_height, curr_bb[2] + delta)
    x0 = max(0, curr_bb[1] - delta)
    x1 = min(max_width, curr_bb[3] + delta)
    subs_image.append(first_image[y0: y1, x0: x1])
    subs_frames_indices.append(curr_subs_frame_indices)

for index, row in df_annotations_in.iterrows():
    if row['subs']:
        image_index = row[0]
        coloured_image = read_image(row['filename'])
        image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
        start_of_frame = True
        if old_index == image_index - 1 and not curr_bb is None:
            image_zone = first_bw_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]]
            curr_zone = np.array(image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
            if same_subs(curr_zone, image_zone):
                image_bb = np.array([row['y0'], row['x0'],
                                     row['y1'], row['x1']])
                curr_bb = np.array([min(curr_bb[0], image_bb[0]), min(curr_bb[1], image_bb[1]),
                                    max(curr_bb[2], image_bb[2]), max(curr_bb[3], image_bb[3])])
                curr_subs_frame_indices.append(image_index)
                start_of_frame = False
                old_index = image_index
            else:
                finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)
                start_of_frame = True
        if start_of_frame:
            curr_subs_frame_indices = [image_index]
            old_index = image_index
            first_image = coloured_image
            first_bw_image = image
            curr_bb = np.array([row['y0'], row['x0'],
                                row['y1'], row['x1']])
    elif len(curr_subs_frame_indices) > 0:
        finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)

if not curr_subs_frame_indices is None:
    finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)

filenames = []
for i, elt in enumerate(subs_image):
    filename = f"{directory}/{prefix}-extraction-{i}.jpg"
    filenames.append(filename)
    cv2.imwrite(filename, elt)

df_subs_group = pd.DataFrame(data={
    'frames': subs_frames_indices,
    'filename': filenames
})

df_subs_group.to_csv(f"{directory}/{prefix}-extraction.csv", encoding='utf-8')
