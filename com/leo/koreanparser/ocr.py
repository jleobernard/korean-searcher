import io
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision

from com.leo.koreanparser.dl.utils.data_utils import read_image

GOOGLE_MAX_HEIGHT = 2050
GOOGLE_MAX_WIDTH = 1536
NB_ROWS = 1
NB_COLUMNS = 1
COLUMN_HEIGHT = int(GOOGLE_MAX_HEIGHT / NB_ROWS)
COLUMN_WIDTH = int(GOOGLE_MAX_WIDTH / NB_COLUMNS)

class obj:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def get_block_coord(block) -> int:
    bounding_box = block.bounding_box
    first_vertex = bounding_box.vertices[0]
    x0, y0 = first_vertex.x, first_vertex.y
    row = int(math.floor(y0 / COLUMN_HEIGHT))
    column = int(math.floor(x0 / COLUMN_WIDTH))
    return row * NB_COLUMNS + column

def get_texts(document):
    my_texts = [''] * (NB_ROWS * NB_COLUMNS)
    for page in document.pages:
        for block in page.blocks:
            text = []
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if symbol.text:
                            text.append(symbol.text)
            if len(text) > 0:
                block_index = get_block_coord(block)
                my_texts[block_index] = ' '.join(text)
    return my_texts

def send_image_to_google(bg_image_path):
    #google_vision_response_file = "/opt/projetcs/ich/korean-searcher/com/leo/koreanparser/full-text-annotation.json"
    #return json.load(open(google_vision_response_file, 'r'), object_hook=obj).fullTextAnnotation
    client = vision.ImageAnnotatorClient()
    with io.open(bg_image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    print("Calling Google")
    response = client.document_text_detection(image=image)
    print("End calling Google")
    return response.full_text_annotation


annotation_file_name = "/opt/data/korean-subs/work/annotations_okay-not-okay-extraction.csv"
prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
directory = Path(annotation_file_name).resolve().parent
df_annotations_in = pd.read_csv(annotation_file_name)

background_images = []
background_image = np.full((GOOGLE_MAX_HEIGHT, GOOGLE_MAX_WIDTH, 3), 255)

row = 0
column = 0

has_data = False

subtitles_per_frame = []

nb_subs_for_page = 0

for i, annotation in df_annotations_in.iterrows():
    y0 = row * COLUMN_HEIGHT
    x0 = column * COLUMN_WIDTH
    image_file_path = annotation['filename']
    coloured_image = read_image(image_file_path)
    height, width, _ = coloured_image.shape
    ydelta = min(height, COLUMN_HEIGHT)
    xdelta = min(width, COLUMN_WIDTH)
    y1 = y0 + ydelta
    x1 = x0 + xdelta
    background_image[y0:y1, x0: x1, :] = coloured_image[:ydelta, :xdelta, :]
    has_data = True
    column += 1
    nb_subs_for_page += 1
    if column >= NB_COLUMNS:
        column = 0
        row += 1
        if row >= NB_ROWS:
            background_images.append({'image': background_image, 'nb': nb_subs_for_page})
            nb_subs_for_page = 0
            background_image = np.full((GOOGLE_MAX_HEIGHT, GOOGLE_MAX_WIDTH, 3), 255)
            has_data = False
            row = 0
            nb_subs_for_page = 0
if has_data:
    background_images.append({'image': background_image, 'nb': nb_subs_for_page})

for i, bg in enumerate(background_images):
    bg_image_path = f"/tmp/{i}.jpg"
    cv2.imwrite(bg_image_path, bg['image'])
    full_text_annotation = send_image_to_google(bg_image_path)
    subs = get_texts(full_text_annotation)
    for sub in subs[: bg['nb']]:
        subtitles_per_frame.append(sub)

df_annotations_in['subs'] = subtitles_per_frame
df_annotations_in.to_csv(f"{directory}/{prefix}-with-subs.csv", encoding='utf-8')



