from typing import List, Union
from com.leo.koreanparser.model.MultivalueMap import MultivalueMap
from com.leo.koreanparser.model.SearchPattern import SearchPattern
from konlpy.tag import Komoran
import logging
logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

logging.info("Loading Komoran...")
komoran = Komoran()
logging.info("...Komoran loaded")


def match_rule(haystack: str, needle):
    pass


def has_every_word(sentence, words) -> bool:
    return False


def prepare_db(db) -> []:
    return [komoran.pos(sentence) for sentence in db]


def find_positions_by_elements(sentence, search_pattern) -> MultivalueMap[str]:
    return MultivalueMap()


def has_path(multimap_elements: MultivalueMap[str], search_pattern: SearchPattern) -> bool:
    return False


def find_pattern_in_local_db(pattern: str, db: Union[list, List[str]]) -> [str]:
    prepapred_db = prepare_db(db)
    #logging.debug(prepapred_db)
    search_pattern = SearchPattern(pattern, komoran)
    fix_words = search_pattern.get_fix_words()
    candidates: [str] = []
    for sentence in prepapred_db:
        if has_every_word(sentence, fix_words):
            candidates.append(sentence)
    results: [str] = []
    for sentence in candidates:
        multimap_elements = find_positions_by_elements(sentence, search_pattern)
        if has_path(multimap_elements, search_pattern):
            results.append(sentence)
    return results
