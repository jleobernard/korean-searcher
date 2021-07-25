from typing import List, Union, Tuple
from com.leo.koreanparser.model.SearchPattern import SearchPattern, Sentence, Lemme, StateTransition
from konlpy.tag import Komoran
import logging
logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

logging.info("Loading Komoran...")
komoran = Komoran()
logging.info("...Komoran loaded")


def has_every_word(sentence: Sentence, words: [Lemme]) -> bool:
    for content, _ in words:
        found = False
        for content_sentence, _ in sentence:
            if content_sentence == content:
                found = True
                break
        if not found:
            return False
    return True


def prepare_db(db):
    return [(sentence, komoran.pos(sentence)) for sentence in db]


def find_pattern_in_local_db(pattern: str, db: Union[list, List[str]]) -> [str]:
    prepapred_db = prepare_db(db)
    search_pattern = SearchPattern(pattern, komoran)
    fix_words = search_pattern.get_fix_words()
    candidates: [str] = []
    for sentence, prepared_sentence in prepapred_db:
        if has_every_word(prepared_sentence, fix_words):
            candidates.append((sentence, prepared_sentence))
    results: [str] = []
    for sentence, prepared_sentence in candidates:
        # La boucle suivante peut-être optimisée pour savoir à quel index on
        # pourrait reprendre après avoir arrêté à un certain état de la machine
        # à état mais 1/ c'est long à faire 2/ pas sûr qu'on ait de meilleurs
        # résultats comme les phrases et les requêtes sont relativement petites
        for i in range(0, len(prepared_sentence)):
            if search_pattern.matches(prepared_sentence, i):
                results.append(sentence)
                break
    return results
