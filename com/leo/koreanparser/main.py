import string
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from dotenv import load_dotenv



class IncomingVideoFileWatcher:

    def __init__(self, in_directory: string):
        self.observer = Observer()
        self.in_directory = in_directory

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

    @staticmethod
    def on_any_event(event):
        print(event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Démarrage du pipeline d'extraction de sous-titres")
    parser.add_argument('--conf', dest='conf_path', help='Path to conf', required=True)
    args = parser.parse_args()
    load_dotenv(parser.conf_path)
    income_dir = os.getenv("income_dir")
    model_dir = os.getenv("model_dir")
    watcher = IncomingVideoFileWatcher(income_dir)
    watcher.run()