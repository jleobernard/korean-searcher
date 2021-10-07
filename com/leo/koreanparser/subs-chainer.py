import cv2
import numpy as np
import pandas as pd
from com.leo.koreanparser.dl.utils.data_utils import read_image


def same_subs(expected_zone: np.ndarray, image_zone: np.ndarray) -> bool:

    cv2.imwrite('/tmp/tmp-0.jpg', expected_zone)
    cv2.imwrite('/tmp/tmp-1.jpg', image_zone)

    threshold = 200
    expected_zone_thresholded = np.where(expected_zone > threshold, 1, 0)
    image_zone_thresholded = np.where(image_zone > threshold , 1, 0)
    ratio = (expected_zone_thresholded * image_zone_thresholded).sum() / expected_zone_thresholded.sum()
    return ratio > 0.75


df = pd.read_csv("/opt/data/korean-subs/work/annotations_okay-not-okay.csv")
curr_bb = None
first_image = None
first_bw_image = None
subs_frames = []
subs = []
curr_subs_frame = None
old_index = -1
for index, row in df.iterrows():
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
                curr_subs_frame.append(image_index)
                start_of_frame = False
                old_index = image_index
            else:
                subs.append(first_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
                subs_frames.append(curr_subs_frame)
                start_of_frame = True
        if start_of_frame:
            curr_subs_frame = [image_index]
            old_index = image_index
            first_image = coloured_image
            first_bw_image = image
            curr_bb = np.array([row['y0'], row['x0'],
                                row['y1'], row['x1']])
            cv2.imwrite('/tmp/tmp.jpg', curr_bb)
    elif len(curr_subs_frame) > 0:
        subs_frames.append(curr_subs_frame)

if not curr_subs_frame is None:
    subs_frames.append(curr_subs_frame)
    cv2.imwrite('/tmp/tmp.jpg', first_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
print(subs_frames)
