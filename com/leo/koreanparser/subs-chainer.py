import cv2
import numpy as np
import pandas as pd

from com.leo.koreanparser.dl.utils.data_utils import read_image, same_subs

df = pd.read_csv("/opt/data/korean-subs/work/annotations_okay-not-okay.csv")
with_subs = df[df['subs']]
curr_bb = None
first_image = None
subs_frames = []
subs = []
curr_subs_frame = None
for index, row in with_subs.iterrows():
    image_index = row[0]
    coloured_image = read_image(row['filename'])
    image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
    start_of_frame = True
    if not curr_bb is None:
        image_zone = image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]]
        curr_zone = np.array(image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
        if same_subs(curr_zone, image_zone):
            image_bb = np.array([row['y0'], row['x0'],
                                 row['y1'], row['x1']])
            curr_bb = np.array([min(curr_bb[0], image_bb[0]), min(curr_bb[1], image_bb[1]),
                                max(curr_bb[2], image_bb[2]), max(curr_bb[3], image_bb[3])])
            curr_subs_frame.append(image_index)
            start_of_frame = False
        else:
            subs.append(first_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
            cv2.imwrite('/tmp/tmp.jpg', cv2.cvtColor(first_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]], cv2.COLOR_RGB2BGR))
            subs_frames.append([curr_subs_frame])
            start_of_frame = True
    if start_of_frame:
        curr_subs_frame = [image_index]
        first_image = coloured_image
        curr_bb = np.array([row['y0'], row['x0'],
                            row['y1'], row['x1']])
        cv2.imwrite('/tmp/tmp.jpg', curr_bb)
if not curr_subs_frame is None:
    subs_frames.append([curr_subs_frame])
    cv2.imwrite('/tmp/tmp.jpg', first_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
print(curr_subs_frame)
