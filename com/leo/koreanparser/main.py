import string
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, EVENT_TYPE_CLOSED
import os
from dotenv import load_dotenv
import cv2


class IncomingVideoFileWatcher:

    def __init__(self):
        self.observer = Observer()

    def run(self):
        in_directory = os.getenv("in_directory")
        work_directory = os.getenv("work_directory")
        skip_frames = os.getenv("skip_frames")
        event_handler = Handler(work_directory=work_directory, skip_frames=skip_frames)
        self.observer.schedule(event_handler, in_directory, recursive=False)
        self.observer.start()
        print(f"Watching     directory {in_directory}")
        print(f"Working with directory {work_directory}")
        try:
            while True:
                time.sleep(5)
        except:
            print("Error")
        finally:
            self.observer.stop()
            self.observer.join()


class Handler(FileSystemEventHandler):

    def __init__(self, work_directory: string, skip_frames: int = 30):
        self.work_directory = work_directory
        self.skip_frames = skip_frames
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
                print("Erreur")

    def treat_incoming_file(self, file_path):
        print(f"Treating file {file_path}")
        self.split_file(file_path)

    def split_file(self, file_path):
        print(f"Splitting file {file_path}")
        cap = cv2.VideoCapture(file_path)
        i = 0
        stop_all = False
        while cap.isOpened() and not stop_all:
            for j in range(0, self.skip_frames):
                ret, frame = cap.read()
                if ret == False:
                    stop_all = True
            if not stop_all:
                cv2.imwrite(f"{self.work_directory}/{args.prefix}-{str(i)}.jpg", frame)
                i += 1
                if i % 10 == 0:
                    print(f"      Exported {i} frames")
        cap.release()
        cv2.destroyAllWindows()
        print(f"End splitting file {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    args = parser.parse_args()
    load_dotenv(args.conf_path)
    watcher = IncomingVideoFileWatcher()
    watcher.run()