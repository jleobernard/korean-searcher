import string
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, EVENT_TYPE_CLOSED
import os
from dotenv import load_dotenv
import cv2
import sys, traceback, shutil



class IncomingVideoFileWatcher:

    def __init__(self):
        self.observer = Observer()

    def run(self):
        in_directory = os.getenv("income_dir")
        work_directory = os.getenv("work_directory")
        skip_frames = os.getenv("skip_frames")
        event_handler = Handler(work_directory=work_directory, skip_frames=skip_frames)
        self.observer.schedule(event_handler, in_directory, recursive=False)
        self.observer.start()
        print(f"Watching     directory {in_directory}")
        print(f"Working with directory {work_directory}")
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

    def __init__(self, work_directory: string, skip_frames: int = 30):
        self.work_directory = work_directory
        self.skip_frames = int(skip_frames)
        self.ensure_dir(work_directory)
        print(f"Work dir is {work_directory}")

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_any_event(self, event: FileSystemEvent):
        ready_file_path = event.src_path
        if not event.is_directory and event.event_type == EVENT_TYPE_CLOSED and ready_file_path and ready_file_path.endswith('.ready'):
            file_path = ready_file_path[:-6]
            os.remove(ready_file_path)
            try:
                self.treat_incoming_file(file_path)
            except:
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)

    def treat_incoming_file(self, file_path):
        print(f"Clean working directory {self.work_directory}")
        with os.scandir(self.work_directory) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
        print(f"Treating file {file_path}")
        self.split_file(file_path)
        time.sleep(30)

    def split_file(self, file_path):
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
                    print(f"      Exported {i} frames")
        cap.release()
        cv2.destroyAllWindows()
        print(f"End splitting file {file_path} into {self.work_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    args = parser.parse_args()
    load_dotenv(args.conf_path)
    watcher = IncomingVideoFileWatcher()
    watcher.run()