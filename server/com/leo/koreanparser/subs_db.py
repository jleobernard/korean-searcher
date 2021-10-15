import logging
import os

import pandas as pd

from com.leo.koreanparser.bo.search_pattern import Lemme

NB_NON_PARSED_COLUMNS = 3

class SubsDbEntry:

    def __init__(self, video_name: str, from_ts: float, to_ts: float, subs: str, tags: [Lemme]):
        self.video_name = video_name
        self.subs = subs
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tags = tags

    def __str__(self) -> str:
        return f"video={self.video_name};{self.from_ts}->{self.to_ts};tags={self.tags}"

class SubsDb:

    def __init__(self):
        self.entries: [SubsDbEntry] = []

    def __str__(self) -> str:
        return '\n'.join([e.__str__() for e in self.entries])

    def load(self, store_path: str):
        logging.info(f"Loading {store_path}")
        for dir_entry in os.listdir(store_path):
            dir_entry_path = f"{store_path}/{dir_entry}"
            if os.path.isdir(dir_entry_path):
                self.__load_dir__(dir_entry_path)
        logging.info("Done loading")
        logging.info(self)

    def __load_dir__(self, dir_entry):
        prefix = os.path.basename(dir_entry)
        csv_file_path = f"{dir_entry}/{prefix}.csv"
        if not os.path.exists(csv_file_path):
            logging.info(f"Not loading {dir_entry} as it doesn't contain annotations")
            return
        logging.info(f"Loading {csv_file_path}...")
        df: pd.DataFrame = pd.read_csv(csv_file_path)
        for i, row in df.iterrows():
            self.entries.append(SubsDbEntry(prefix, row['start'], row['end'], row['subs'],
                                            self.__get_tags_from_row__(row)))
        logging.info(f"...{csv_file_path} loaded")

    def __get_tags_from_row__(self, row) -> [Lemme]:
        tags: [Lemme] = []
        i = 0
        while f"parsed_{i}" in row:
            p1 = row[f"parsed_{i}"]
            if pd.isna(p1):
                break
            tags.append((p1, row[f"parsed_{i + 1}"]))
            i += 2
        return tags

