import string
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, EVENT_TYPE_CLOSED
import os
from dotenv import load_dotenv



class IncomingVideoFileWatcher:

    def __init__(self, in_directory: string, work_directory: string):
        self.observer = Observer()
        self.in_directory = in_directory
        self.work_directory = work_directory

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.in_directory, recursive=False)
        self.observer.start()
        print(f"Watching directory {self.in_directory}")
        try:
            while True:
                time.sleep(5)
        except:
            print("Error")
        finally:
            self.observer.stop()
            self.observer.join()


class Handler(FileSystemEventHandler):

    def on_any_event(self, event: FileSystemEvent):
        ready_file_path = event.src_path
        if not event.is_directory and event.event_type == EVENT_TYPE_CLOSED and ready_file_path and ready_file_path.endswith('.ready'):
            file_path = ready_file_path[:-6]
            print(f"Treating file {file_path}")
            os.remove(ready_file_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    args = parser.parse_args()
    load_dotenv(args.conf_path)
    income_dir = os.getenv("income_dir")
    work_dir = os.getenv("work_dir")
    model_dir = os.getenv("model_dir")
    watcher = IncomingVideoFileWatcher(income_dir, work_directory=work_dir)
    watcher.run()