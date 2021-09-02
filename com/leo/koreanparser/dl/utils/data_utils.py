import argparse
import random
import os
import shutil
import time
from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from com.leo.koreanparser.dl.conf import TARGET_WIDTH, TARGET_HEIGHT
from com.leo.koreanparser.dl.utils.image_helper import normalize_imagenet

IMAGE_EXTENSIONS = ["jpg", "png"]
CSV_ANNOTATION_COL_NAMES = ["label", "x0", "y0", "x1", "y1", "filename", "width", "height"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', dest='datadir',
                        help='path to input images')
    parser.add_argument('-b', '--batch-size', dest='batch_size',
                        default=10, help='batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', default=5,
                        help='path to input images')
    parser.add_argument('--load', dest='load', default=False,
                        help='Load model if true')
    parser.add_argument('--models', dest='models_path',
                        help='Path to the folder containing the models (load and save)', required=True)
    parser.add_argument('--working-dir', dest="working_dir",
                        help='Path to the folder where we can write stuff', required=True)
    parser.add_argument('--lr', dest='lr', default=0.0001,
                        help='Learning rate')
    parser.add_argument('--max-lr', dest='max_lr', default=0.1,
                        help='Max learning rate')
    parser.add_argument('-t', '--threshold', dest='threshold', default=0.5,
                        help='Min score to consider the pciture as containing subtitles')
    parser.add_argument('--alpha', dest='alpha', default=1.0,
                        help='Weight for presence/absence loss')
    parser.add_argument('--beta', dest='beta', default=1.0,
                        help='Weight for center spread loss')
    parser.add_argument('--gamma', dest='gamma', default=1.0,
                        help='Weights for ratio loss')
    parser.add_argument('--theta', dest='theta', default=1.0,
                        help='Weights for difference of dimension loss')
    return vars(parser.parse_args())


class SubsDataset(Dataset):

    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = getNominalData(path, self.bb[idx])
        # x, y_bb = transformsXY(path, self.bb[idx], self.transforms)
        x = normalize_imagenet(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb


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
        annotation_data['subs'] = 1.
        del annotation_data['label']
        annotations_with_subs.append(annotation_data)
    concatenated_data = pd.concat(annotations_with_subs)
    unsubbed = pd.DataFrame(columns=['filename'], data=[os.path.realpath(f) for f in all_images])
    subbed_filenames = concatenated_data[['filename']].copy()
    unsubbed = unsubbed[~unsubbed.filename.isin(subbed_filenames.filename)]
    unsubbed[['x0', 'y0', 'x1', 'y1']] = 0
    unsubbed[['width', 'height']] = unsubbed['filename'].apply(lambda x: Image.open(x).size).tolist()
    unsubbed['subs'] = 0.
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

def resize_image_bb(read_path,write_path,bb):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (TARGET_WIDTH, TARGET_HEIGHT))
    #plt.imshow(im_resized)
    #plt.show()
    Y_resized = cv2.resize(create_mask(bb, im), (TARGET_WIDTH, TARGET_HEIGHT))
    #plt.imshow(Y_resized, cmap='gray')
    #plt.show()
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

def getNominalData(path, bb):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    return x, bb

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

def augment_dataset(df_train):
    print("Create fake input files")
    augmented_dataset = pd.DataFrame(columns=df_train.columns)
    for index, row in df_train.iterrows():
        # Flip along Y axis
        path_to_file = row['new_path']
        bb = row['new_bb']
        new_path = path_to_file + '.augmented.jpg'
        im = read_image(path_to_file)
        flipped_image = cv2.flip(im, 1)
        cv2.imwrite(new_path, cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR))
        augmented_dataset = augmented_dataset.append({
            'subs': row['subs'],
            'filename': row['filename'],
            'new_path': new_path,
            'new_bb': np.array([bb[0], 1 - bb[3], bb[2], 1 - bb[1]], dtype=np.float32)
        }, ignore_index=True)
    return augmented_dataset


def load_train_data(path, working_dir_path: str):
    print("Loading data....")
    df_train = generate_train_df(path)
    new_paths = []
    new_bbs = []
    train_path_resized = Path(working_dir_path)
    Path(train_path_resized).mkdir(parents=True, exist_ok=True)
    clean_dir(train_path_resized)
    not_found: [str] = []
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    for index, row in df_train.iterrows():
        filename = row['filename']
        try:
            new_path, new_bb = resize_image_bb(filename, train_path_resized, create_bb_array(row.values))
            new_paths.append(new_path)
            new_bbs.append(new_bb) # / np.array([TARGET_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH]))
        except:
            print(f"Could not open file {filename}")
            not_found.append(filename)
    if len(not_found) > 0:
        df_train = df_train[~df_train.filename.isin(not_found)]
    df_train['new_path'] = new_paths
    df_train['new_bb'] = new_bbs
    #augmented = augment_dataset(df_train)
    #show_sample_image(df_train)
    #show_sample_image(augmented)
    print("...data loaded")
    #return df_train.append(augmented)
    return df_train

def show_sample_image(df_train: pd.DataFrame):
    #return
    im = cv2.imread(str(df_train.values[2][-2]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    show_corner_bb(im, df_train.values[2][-1])
    input("Press Enter to continue.")


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    height, width, _ = im.shape
    resized_bb = bb * np.array([height, width, height, width])
    plt.gca().add_patch(create_corner_rect(resized_bb))
    plt.show()
