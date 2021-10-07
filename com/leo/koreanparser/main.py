import string
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, EVENT_TYPE_CLOSED, EVENT_TYPE_MODIFIED
import os
from dotenv import load_dotenv
import cv2
import sys, traceback, shutil
import pandas as pd
import numpy as np
import torch
from com.leo.koreanparser.dl.conf import TARGET_HEIGHT, TARGET_WIDTH
from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.predict import get_bb_from_bouding_boxes

from com.leo.koreanparser.dl.utils.data_utils import read_image, SubsDataset
from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device
from com.leo.koreanparser.dl.utils.train_utils import do_lod_specific_model

class IncomingVideoFileWatcher:

    def __init__(self):
        self.observer = Observer()

    def run(self):
        in_directory = os.getenv("income_dir")
        work_directory = os.getenv("work_directory")
        skip_frames = os.getenv("skip_frames")
        model_path = os.getenv("model_path")
        event_handler = Handler(work_directory=work_directory, skip_frames=skip_frames, weights_path=model_path)
        self.observer.schedule(event_handler, in_directory, recursive=False)
        self.observer.start()
        print(f"Watching     directory {in_directory}")
        print(f"Working with directory {work_directory}")
        print(f"Loaded model           {model_path}")
        try:
            while True:
                time.sleep(100)
        except:
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
        finally:
            self.observer.stop()
            self.observer.join()


class Handler(FileSystemEventHandler):

    def __init__(self, work_directory: string, weights_path: str, skip_frames: int = 30):
        self.work_directory = work_directory
        self.skip_frames = int(skip_frames)
        self.ensure_dir(work_directory)
        self.threshold = 0.5
        self.model = get_model(eval=True)
        self.treated = set([])
        do_lod_specific_model(weights_path, self.model)

    def ensure_dir(self, file_path):
        os.makedirs(file_path, exist_ok=True)

    def on_any_event(self, event: FileSystemEvent):
        ready_file_path = event.src_path
        if not event.is_directory and \
                (event.event_type == EVENT_TYPE_CLOSED or event.event_type == EVENT_TYPE_MODIFIED) \
                and ready_file_path and ready_file_path.endswith('.ready') and not ready_file_path in self.treated:
            self.treated.add(ready_file_path)
            file_path = ready_file_path[:-6]
            try:
                self.treat_incoming_file(file_path)
            except:
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
            finally:
                os.remove(ready_file_path)

    def treat_incoming_file(self, file_path):
        print(f"Clean working directory {self.work_directory}")
        with os.scandir(self.work_directory) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
        print(f"Treating file {file_path}")
        prefix_splitted: str = self.split_file(file_path)
        annotation_file, annotations = self.create_annotations(prefix_splitted)
        #self.extract_bounding_rects =

    def split_file(self, file_path) -> str:
        print(f"Splitting file {file_path}")
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
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
        size = (TARGET_WIDTH, TARGET_HEIGHT)
        data = []
        i = 0
        while True:
            splitted_file = f"{self.work_directory}/{prefix_splitted}-{i}.jpg"
            if os.path.exists(splitted_file):
                print(f"--- Reading splitted file {splitted_file}")
                im = read_image(splitted_file)
                print(f"------ Resizing")
                im = cv2.resize(im, size)
                resized_file_path = f"{self.work_directory}/{prefix_splitted}-resized-{i}.jpg"
                cv2.imwrite(resized_file_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                test_ds = SubsDataset(pd.DataFrame([{'path': resized_file_path}])['path'], pd.DataFrame([{'bb': np.array([0, 0, 0, 0])}])['bb'], pd.DataFrame([{'y': [0]}])['y'])
                x, y_class, y_bb, _ = test_ds[0]
                xx = to_best_device(torch.FloatTensor(x[None, ]))
                print(f"------ Inference")
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
                print(f"--- Splitted file {splitted_file} treated")
                os.remove(splitted_file)
            else:
                print(f"--- Splitted file {splitted_file} does not exist so we will stop")
                break
            i += 1
        annotations = pd.DataFrame(columns=['filename', 'subs', 'x0', 'y0', 'x1', 'y1', 'p'], data=data)
        annotations_file_path = f"{self.work_directory}/annotations_{prefix_splitted}.csv"
        annotations.to_csv(annotations_file_path, encoding='utf-8')
        return annotations_file_path, annotations




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    args = parser.parse_args()
    load_dotenv(args.conf_path)
    watcher = IncomingVideoFileWatcher()
    watcher.run()