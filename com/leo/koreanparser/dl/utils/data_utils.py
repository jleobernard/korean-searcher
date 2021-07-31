import argparse
import random
import os
import shutil
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame

IMAGE_EXTENSIONS = ["jpg", "png"]
CSV_ANNOTATION_COL_NAMES = ["label", "x0", "y0", "x1", "y1", "filename", "width", "height"]


class CompleteAnnotationData:

    def __init__(self, filename: str, has_sub: bool, width: int, height: int, xmin: int, ymin: int, xmax: int, ymax: int):
        self.filename = filename
        self.has_sub = has_sub
        self.width = width
        self.height = height
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def get_last_model_params(models_rep) -> Union[str, None]:
    if not os.path.exists(models_rep):
        os.makedirs(models_rep)
    else:
        file_list = os.listdir(models_rep)
        file_list = [f for f in file_list if f[-3:] == '.pt']
        file_list.sort(reverse=True)
        if len(file_list) > 0:
            return f"{models_rep}/{file_list[0]}"
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--data', dest='data_path',
                        help='Path to the folder containing training data', required=True)
    parser.add_argument('--models', dest='models_path',
                        help='Path to the folder containing the models (load and save)', required=True)
    parser.add_argument('--epoch', dest='epoch', default=10,
                        help='Path to the folder containing training data')
    parser.add_argument('--batch', dest='batch', default=10,
                        help='Number of images per batch')
    parser.add_argument('--sentence', dest='sentence', default=10,
                        help='Max length of sentences')
    parser.add_argument('--lr', dest='lr', default=0.0001,
                        help='Learning rate')
    parser.add_argument('--max-lr', dest='max_lr', default=0.1,
                        help='Max learning rate')
    parser.add_argument('--load', dest='load', default=False,
                        help='Load model if true')
    parser.add_argument('--feat-mul', dest='feat_mul', default=15,
                        help='Load model if true')
    parser.add_argument('--val-freq', dest='val_freq', default=5,
                        help='Validation frequency')
    parser.add_argument('--val-patience', dest='val_patience', default=5,
                        help='Validation patience')
    return parser.parse_args()


def get_file_extension(file_path):
    extension = os.path.splitext(file_path)[1]
    return extension[1:] if extension else extension


def list_files(data_dir, file_types: Union[str, List[str]]):
    """Returns a fully-qualified list of filenames under root directory"""
    _fts = [file_types] if isinstance(file_types, str) else file_types
    return [os.path.join(data_dir, f) for data_dir, directory_name,
                    files in os.walk(data_dir) for f in files if get_file_extension(f) in _fts]


def generate_train_df (data_dir):
    """
    :param data_dir: Where to look for data
    :return:
    """
    annotations_with_subs: [pd.DataFrame] = []
    annotation_files = list_files(data_dir, "csv")
    all_images = list_files(data_dir, IMAGE_EXTENSIONS)
    for annotation_file in annotation_files:
        annotation_dir_path = os.path.dirname(os.path.realpath(annotation_file))
        annotation_data: DataFrame = pd.read_csv(annotation_file, names=CSV_ANNOTATION_COL_NAMES)
        annotation_data.filename = annotation_dir_path + "/" + annotation_data.filename
        annotation_data['x1'] = annotation_data['x0'] + annotation_data['x1']
        annotation_data['y1'] = annotation_data['y0'] + annotation_data['y1']
        annotation_data['subs'] = True
        del annotation_data['label']
        annotations_with_subs.append(annotation_data)
    concatenated_data = pd.concat(annotations_with_subs)
    unsubbed = pd.DataFrame(columns=['filename'], data=all_images)
    subbed_filenames = concatenated_data[['filename']].copy()
    unsubbed = unsubbed[~unsubbed.filename.isin(subbed_filenames.filename)]
    unsubbed[['x0', 'y0', 'x1', 'y1']] = 0
    unsubbed[['width', 'height']] = unsubbed['filename'].apply(lambda x: Image.open(x).size).tolist()
    unsubbed['subs'] = False
    df_train = pd.concat([concatenated_data, unsubbed])
    df_train['filename'] = df_train['filename'].apply(lambda x: Path(x))
    return df_train


def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y


def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[1], x[0], x[3], x[2]])

def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# modified from fast.ai
def crop(im, r, c, target_r, target_c):
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        if np.random.random() > 0.5:
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)


def load(path):
    df_train = generate_train_df(path)
    new_paths = []
    new_bbs = []
    train_path_resized = Path(f'{path}/resized')
    Path(train_path_resized).mkdir(parents=True, exist_ok=True)
    clean_dir(train_path_resized)
    for index, row in df_train.iterrows():
        new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values), 400)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    df_train['new_path'] = new_paths
    df_train['new_bb'] = new_bbs
    return df_train

df_train = load("/opt/projetcs/ich/korean-searcher/com/leo/koreanparser/dl/data/input")
nb_subbed = len(df_train[df_train.subs])
nb_unsubbed = len(df_train[~df_train.subs])
print(f"There are {nb_subbed} files with subtitles and {nb_unsubbed} files without for a total of {nb_unsubbed + nb_subbed} files")



