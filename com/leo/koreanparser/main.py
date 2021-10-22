import argparse
import io
import itertools
import json
import logging
import math
import os
import string

import cv2
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from google.cloud import vision
from konlpy.tag import Komoran

from com.leo.koreanparser.dl.conf import TARGET_HEIGHT, TARGET_WIDTH
from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.predict import get_bb_from_bouding_boxes
from com.leo.koreanparser.dl.utils.data_utils import read_image, SubsDataset
from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device
from com.leo.koreanparser.dl.utils.train_utils import do_lod_specific_model

logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

GOOGLE_MAX_HEIGHT = 2050
GOOGLE_MAX_WIDTH = 1536
NB_ROWS = 10
NB_COLUMNS = 3
COLUMN_HEIGHT = int(GOOGLE_MAX_HEIGHT / NB_ROWS)
COLUMN_WIDTH = int(GOOGLE_MAX_WIDTH / NB_COLUMNS)

BREAKS = vision.TextAnnotation.DetectedBreak.BreakType

class obj:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


class Handler():

    def __init__(self, work_directory: string, weights_path: str, target_directory: str, skip_frames: int = 30):
        self.work_directory = work_directory
        self.target_directory = target_directory
        self.skip_frames = int(skip_frames)
        self.ensure_dir(work_directory)
        self.ensure_dir(target_directory)
        self.threshold = 0.5
        self.model = get_model(eval=True)
        self.treated = set([])
        do_lod_specific_model(weights_path, self.model)
        logging.info("Loading Komoran...")
        self.analyzer = Komoran()
        logging.info("...Komoran loaded")
        logging.info(f"Working with directory {work_directory}")
        logging.info(f"Target       directory {target_directory}")
        logging.info(f"Loaded model           {weights_path}")

    def ensure_dir(self, file_path):
        os.makedirs(file_path, exist_ok=True)

    def treat_incoming_file(self, file_path):
        logging.info(f"Treating file {file_path}")
        work_file_path = self.prepare_file(file_path)
        if work_file_path:
            prefix_splitted: str = self.split_file(work_file_path)
            annotation_file = self.create_annotations(prefix_splitted)
            annotation_file_extraction = self.extract_bounding_rects(annotation_file)
            annotation_file_with_subs = self.add_subs(annotation_file_extraction, prefix_splitted)
            annotation_file_polished = self.polish(annotation_file_with_subs, prefix_splitted, work_file_path)
            annotation_file_final = self.parse_korean(annotation_file_polished, prefix_splitted)
            self.move_products(prefix_splitted, [annotation_file_final, work_file_path, file_path])
            logging.info(f"Done ! {file_path} treated")

    def move_products(self, prefix: str, products_path: [str]):
        logging.info(f"Moving products of {prefix} to {self.target_directory}")
        dest_dir = f"{self.target_directory}/{prefix}"
        self.ensure_dir(dest_dir)
        for path in products_path:
            file_name = os.path.basename(path)
            os.rename(path, f"{dest_dir}/{file_name}")
            logging.info(f"------ {path} moved to {dest_dir}/{file_name}")

    def parse_korean(self, annotation_file_path: str, prefix: str) -> str:
        logging.info("Parsing korean subs")
        final_annotation_file_path = f"{self.work_directory}/{prefix}.csv"
        df_in = pd.read_csv(annotation_file_path)
        extends = []
        nb_lines = len(df_in)
        for i, row in df_in.iterrows():
            parsed = self.analyzer.pos(row['subs'])
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
        df_in.to_csv(final_annotation_file_path, encoding='utf-8')
        return final_annotation_file_path

    def polish(self, annotation_file_path: str, prefix: str, video_file_path: str):
        print("Polishing of the file")
        final_annotation_file_path = f"{self.work_directory}/{prefix}-polished.csv"
        if os.path.exists(final_annotation_file_path):
            print("--- Polishing already done")
        else:
            df_in = pd.read_csv(annotation_file_path)
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            spf = 1 / fps
            spf_for_sampling_rate = self.skip_frames * spf

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
            df.to_csv(final_annotation_file_path)
        return final_annotation_file_path


    def prepare_file(self, file_path: str) -> str:
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        out_file = f"{self.work_directory}/{filename_without_extension}.mp4"
        if os.path.exists(out_file):
            print("File already converted")
            return out_file
        else:
            print(f"Converting {file_path} into {out_file}")
            os.system(f"ffmpeg -i {file_path} {out_file}")
            if os.path.isfile(out_file):
                print(f"Conversion successful")
                return out_file
            else:
                print(f"Could not convert file")
                return None

    def split_file(self, file_path) -> str:
        print(f"Splitting file {file_path}")
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        if os.path.exists(f"{self.work_directory}/{filename_without_extension}-0.jpg"):
            print(f"--- File already splitted")
        else:
            cap = cv2.VideoCapture(file_path)
            i = 0
            stop_all = False
            while cap.isOpened() and not stop_all:
                for j in range(0, self.skip_frames):
                    ret, frame = cap.read()
                    if ret == False:
                        stop_all = True
                if not stop_all:
                    cv2.imwrite(f"{self.work_directory}/{filename_without_extension}-{str(i)}.jpg", frame)
                    i += 1
                    if i % 10 == 0:
                        print(f"--- Exported {i} frames")
            cap.release()
            cv2.destroyAllWindows()
            print(f"End splitting file {file_path} into {self.work_directory} with prefix {filename_without_extension}")
        return filename_without_extension

    def create_annotations(self, prefix_splitted: str):
        print("Predicting subs from video")
        annotations_file_path = f"{self.work_directory}/annotations_{prefix_splitted}.csv"
        if os.path.exists(annotations_file_path):
            print("--- subs prediction already done")
        else:
            size = (TARGET_WIDTH, TARGET_HEIGHT)
            data = []
            i = 0
            while True:
                splitted_file = f"{self.work_directory}/{prefix_splitted}-{i}.jpg"
                if os.path.exists(splitted_file):
                    im = read_image(splitted_file)
                    im = cv2.resize(im, size)
                    resized_file_path = f"{self.work_directory}/{prefix_splitted}-resized-{i}.jpg"
                    cv2.imwrite(resized_file_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                    test_ds = SubsDataset(pd.DataFrame([{'path': resized_file_path}])['path'], pd.DataFrame([{'bb': np.array([0, 0, 0, 0])}])['bb'], pd.DataFrame([{'y': [0]}])['y'])
                    x, y_class, y_bb, _ = test_ds[0]
                    xx = to_best_device(torch.FloatTensor(x[None, ]))
                    out_class, out_bb = self.model(xx)
                    class_hat = torch.sigmoid(out_class.detach().cpu()).numpy()
                    if class_hat[0][0] >= self.threshold:
                        bb_hat = out_bb.detach().cpu()
                        bounding_boxes = get_bb_from_bouding_boxes(bb_hat, height=TARGET_HEIGHT, width=TARGET_WIDTH)
                        bb = bounding_boxes[0].numpy()
                        y0 = int(np.floor(min(bb[0], bb[2])))
                        y1 = int(np.ceil(max(bb[0], bb[2])))
                        x0 = int(np.floor(min(bb[1], bb[3])))
                        x1 = int(np.ceil(max(bb[1], bb[3])))
                        data.extend([[resized_file_path, True, x0, y0, x1, y1, class_hat[0][0]]])
                    else:
                        data.extend([[resized_file_path, False, 0, 0, 0, 0, class_hat[0][0]]])
                    os.remove(splitted_file)
                else:
                    print(f"--- inference done on {i} files")
                    break
                i += 1
            annotations = pd.DataFrame(columns=['filename', 'subs', 'x0', 'y0', 'x1', 'y1', 'p'], data=data)
            annotations.to_csv(annotations_file_path, encoding='utf-8')
        return annotations_file_path

    def same_subs(self, expected_zone: np.ndarray, image_zone: np.ndarray) -> bool:
        threshold = 220
        expected_zone_thresholded = np.where(expected_zone > threshold, 1, 0)
        image_zone_thresholded = np.where(image_zone > threshold , 1, 0)
        ratio = (expected_zone_thresholded * image_zone_thresholded).sum() / (expected_zone_thresholded.sum() + 1e-6)
        return ratio > 0.75

    def finish_frame(self, first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices):
        delta = 10
        max_height, max_width, _ = first_image.shape
        y0 = max(0, curr_bb[0] - delta)
        y1 = min(max_height, curr_bb[2] + delta)
        x0 = max(0, curr_bb[1] - delta)
        x1 = min(max_width, curr_bb[3] + delta)
        subs_image.append(first_image[y0: y1, x0: x1])
        subs_frames_indices.append(curr_subs_frame_indices)

    def extract_bounding_rects(self, annotation_file_name) -> str:
        print("Extract bounding rectangles")
        prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
        df_file_name = f"{self.work_directory}/{prefix}-extraction.csv"
        if os.path.exists(df_file_name):
            print("--- Bounding boxes already extracted")
        else:
            df_annotations_in = pd.read_csv(annotation_file_name)
            curr_bb = None
            first_image = None
            first_bw_image = None
            subs_frames_indices = []
            subs_image = []
            curr_subs_frame_indices = None
            old_index = -1
            for index, row in df_annotations_in.iterrows():
                if row['subs']:
                    image_index = row[0]
                    coloured_image = read_image(row['filename'])
                    image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
                    start_of_frame = True
                    if old_index == image_index - 1 and not curr_bb is None:
                        image_zone = first_bw_image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]]
                        curr_zone = np.array(image[curr_bb[0]: curr_bb[2], curr_bb[1]: curr_bb[3]])
                        if self.same_subs(curr_zone, image_zone):
                            image_bb = np.array([row['y0'], row['x0'],
                                                 row['y1'], row['x1']])
                            curr_bb = np.array([min(curr_bb[0], image_bb[0]), min(curr_bb[1], image_bb[1]),
                                                max(curr_bb[2], image_bb[2]), max(curr_bb[3], image_bb[3])])
                            curr_subs_frame_indices.append(image_index)
                            start_of_frame = False
                            old_index = image_index
                        else:
                            self.finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)
                            start_of_frame = True
                    if start_of_frame:
                        curr_subs_frame_indices = [image_index]
                        old_index = image_index
                        first_image = coloured_image
                        first_bw_image = image
                        curr_bb = np.array([row['y0'], row['x0'],
                                            row['y1'], row['x1']])
                elif curr_subs_frame_indices is not None and len(curr_subs_frame_indices) > 0:
                    self.finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)
                    curr_subs_frame_indices = None

            if not curr_subs_frame_indices is None:
                self.finish_frame(first_image, curr_bb, subs_image, subs_frames_indices, curr_subs_frame_indices)

            filenames = []
            for i, elt in enumerate(subs_image):
                filename = f"{self.work_directory}/{prefix}-extraction-{i}.jpg"
                filenames.append(filename)
                cv2.imwrite(filename, elt)

            df_subs_group = pd.DataFrame(data={
                'frames': subs_frames_indices,
                'filename': filenames
            })
            df_subs_group.to_csv(df_file_name, encoding='utf-8')
        return df_file_name

    def add_subs(self, annotation_file_extraction: str, prefix: str) -> str:
        print("Adding subs")
        file_with_subs_path = f"{self.work_directory}/{prefix}-with-subs.csv"
        if os.path.exists(file_with_subs_path):
            print("--- Subs already added")
        else:
            df_annotations_in = pd.read_csv(annotation_file_extraction)
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
                # On part de en bas à droite et non en haut à gauche pour éviter les problèmes avec les images mal
                # découpées (en bas à droite on a plus de chances de trouver des sous-titres que en haut à gauche dans
                # une image mal découpée)
                background_image[y0:y1, x0: x1, :] = coloured_image[-ydelta:, -xdelta:, :]
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
                bg_image_path = f"{self.work_directory}/{i}.jpg"
                cv2.imwrite(bg_image_path, bg['image'])
                full_text_annotation = self.send_image_to_google(bg_image_path)
                subs = self.get_texts(full_text_annotation)
                for sub in subs[: bg['nb']]:
                    subtitles_per_frame.append(sub)

            df_annotations_in['subs'] = subtitles_per_frame
            df_annotations_in.to_csv(file_with_subs_path, encoding='utf-8')
        return file_with_subs_path

    def send_image_to_google(self, bg_image_path):
        print(f"........... Appel à Google")
        client = vision.ImageAnnotatorClient()
        with io.open(bg_image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        return response.full_text_annotation

    def get_block_coord(self, block) -> int:
        bounding_box = block.bounding_box
        first_vertex = bounding_box.vertices[0]
        x0, y0 = first_vertex.x, first_vertex.y
        row = int(math.floor(y0 / COLUMN_HEIGHT))
        column = int(math.floor(x0 / COLUMN_WIDTH))
        return row * NB_COLUMNS + column

    def get_texts(self, document):
        my_texts = [''] * (NB_ROWS * NB_COLUMNS)
        for page in document.pages:
            for block in page.blocks:
                text = []
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            if symbol.text:
                                if hasattr(symbol, 'property') and \
                                        (hasattr(symbol.property, 'detectedBreak') or
                                         hasattr(symbol.property, 'detected_break')):
                                    if hasattr(symbol.property, 'detectedBreak'):
                                        detected_break = symbol.property.detectedBreak.type
                                    else:
                                        detected_break = symbol.property.detected_break.type_
                                    if detected_break == BREAKS.UNKNOWN:
                                        break_symbol = ''
                                    elif detected_break == 'LINE_BREAK':
                                        break_symbol = '\n'
                                    else:
                                        break_symbol = ' '
                                else:
                                    break_symbol = ''
                                if symbol.text:
                                    text.append(f"{symbol.text}{break_symbol}")
                if len(text) > 0:
                    block_index = self.get_block_coord(block)
                    my_texts[block_index] = ''.join(text)
        return my_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    parser.add_argument('--file', dest='file', help='Path to file to analyze', required=True)
    args = parser.parse_args()
    load_dotenv(args.conf_path)

    in_directory = os.getenv("income_dir")
    work_directory = os.getenv("work_directory")
    skip_frames = os.getenv("skip_frames")
    model_path = os.getenv("model_path")
    target_directory = os.getenv("target_directory")
    handler = Handler(work_directory=work_directory, skip_frames=int(skip_frames), weights_path=model_path,
                      target_directory=target_directory)
    handler.treat_incoming_file(args.file)
