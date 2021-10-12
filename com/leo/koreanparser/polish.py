import io
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision

from com.leo.koreanparser.dl.utils.data_utils import read_image
import json

video_file_name = "/Users/leo/Downloads/okay-not-okay.mp4"
annotation_file_name = "/opt/data/korean-subs/work/test-with-subs.csv"
prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
directory = Path(annotation_file_name).resolve().parent
df_in = pd.read_csv(annotation_file_name)

video = cv2.VideoCapture(video_file_name)
fps = video.get(cv2.CAP_PROP_FPS)
spf = 1 / fps
spf_for_sampling_rate = 30 * spf

# Merge lines with same subs
data = []
curr_subs = None
curr_frame_start, curr_frame_end = None, None
for i, annotation in df_in.iterrows():
    frames = json.loads(annotation['frames'])
    subs = annotation['subs']
    if curr_subs == subs:
        curr_frame_end = frames[-1]
    else:
        if not curr_subs is None:
            data.append([curr_subs, curr_frame_start, curr_frame_end])
        if pd.isna(subs):
            curr_frame_start = None
            curr_frame_end = None
            curr_subs = None
        else:
            curr_frame_start = frames[0]
            curr_frame_end = frames[-1]
            curr_subs = subs
if not curr_subs is None:
    data.append([curr_subs, curr_frame_start, curr_frame_end])
df = pd.DataFrame(columns=['subs', 'start', 'end'], data=data)
df['start'] = df['start'] * spf_for_sampling_rate
df['end'] = df['end'] * spf_for_sampling_rate
df.to_csv(f"{directory}/final-annotations-{prefix}.csv")
