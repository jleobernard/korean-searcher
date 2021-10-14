import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision

from com.leo.koreanparser.dl.utils.data_utils import read_image
from konlpy.tag import Komoran
import logging
import itertools


logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

logging.info("Loading Komoran...")
komoran = Komoran()
logging.info("...Komoran loaded")

annotation_file_name = "/opt/data/korean-subs/work/okay-not-okay-ep02-0-polished.csv"
prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
directory = Path(annotation_file_name).resolve().parent
df_in = pd.read_csv(annotation_file_name)

extends = []
nb_lines = len(df_in)

for i, row in df_in.iterrows():
    parsed = komoran.pos(row['subs'])
    extension = []
    for p in parsed:
        extension.append(p[0])
        extension.append(p[1])
    extends.append(extension)
    if (i + 1) % 100 == 0:
        logging.debug(f"{i + 1} / {nb_lines} subs analyzed")

extends = np.array(list(zip(*itertools.zip_longest(*extends, fillvalue=''))))
nb_extra_columns = len(extends[0])
for i in range(nb_extra_columns):
    df_in[f"parsed_{i}"] = extends[:, i]
print(df_in)
df_in.to_csv(f"{directory}/{prefix}-00.csv", encoding='utf-8')



