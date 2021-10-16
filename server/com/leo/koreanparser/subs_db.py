import logging
import os
import pandas as pd
from konlpy.tag import Komoran
from com.leo.koreanparser.bo.search_pattern import Lemme
from com.leo.koreanparser.bo.search_pattern import SearchPattern

NB_NON_PARSED_COLUMNS = 3

class SubsDbEntryId:
    entry_id = 0

    def get_inc_id(self):
        haidi = self.entry_id
        self.entry_id += 1
        return haidi


ids = SubsDbEntryId()


class SubsDbEntry:

    def __init__(self, video_name: str, from_ts: float, to_ts: float, subs: str, tags: [Lemme]):
        self.id = ids.get_inc_id()
        self.video_name = video_name
        self.subs = subs
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tags = tags

    def __str__(self) -> str:
        return f"video={self.video_name};{self.from_ts}->{self.to_ts};tags={self.tags}"

class SubsDb:

    def __init__(self, analyzer: Komoran):
        self.entries: [SubsDbEntry] = []
        self.analyzer = analyzer

    def __str__(self) -> str:
        return '\n'.join([e.__str__() for e in self.entries])

    def load(self, store_path: str):
        logging.info(f"Loading {store_path}")
        for dir_entry in os.listdir(store_path):
            dir_entry_path = f"{store_path}/{dir_entry}"
            if os.path.isdir(dir_entry_path):
                self.__load_dir__(dir_entry_path)
        logging.info("Done loading")

    def search(self, query: str) -> SubsDbEntry:
        logging.info(f"Looking for {query}")
        search_pattern = SearchPattern(query, self.analyzer)
        fix_words = search_pattern.get_fix_words()
        candidates: [SubsDbEntry] = []
        for entry in self.entries:
            if self.__has_every_word__(entry, fix_words):
                candidates.append(entry)
        results: [SubsDbEntry] = []
        """
        for candidate in candidates:
            # La boucle suivante peut-être optimisée pour savoir à quel index on
            # pourrait reprendre après avoir arrêté à un certain état de la machine
            # à état mais 1/ c'est long à faire 2/ pas sûr qu'on ait de meilleurs
            # résultats comme les phrases et les requêtes sont relativement petites
            for i in range(0, len(prepared_sentence)):
                if search_pattern.matches(prepared_sentence, i):
                    results.append(sentence)
                    break
        """
        logging.info(f"Search for {query} done")
        return candidates


    def __has_every_word__(self, entry: SubsDbEntry, words: [Lemme]) -> bool:
        for content, _ in words:
            found = False
            for content_sentence, _ in entry.tags:
                if content_sentence == content:
                    found = True
                    break
            if not found:
                return False
        return True

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

