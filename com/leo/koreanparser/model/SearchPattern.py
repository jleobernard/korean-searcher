from __future__ import annotations
from typing import Tuple
from konlpy.tag import Komoran


def get_kw_and_replacement(keyword: str):
    if keyword == '(':
        replacement = "STARTCONDITIONAL"
    elif keyword == ')':
        replacement = "ENDCONDITIONAL"
    else:
        replacement = keyword[1:-1] + "MMMMMMMMMMM"
    return keyword, replacement

def get_kw_from_replacement(replacement: str):
    return [kw for kw, rpl in KEYWORDS_AND_REPLACEMENT if rpl == replacement][0]


KEYWORDS = ['<VSTEM>', '<ADJSTEM>', '≤NSTEM>', '<WORDS>', '<WORD>', '(', ')']
KEYWORDS_AND_REPLACEMENT = [get_kw_and_replacement(kw) for kw in KEYWORDS]
POS_TAG_NOT_WORD = ['SC', 'SE', 'SF', 'SSC', 'SSO']


class State:
    def __init__(self, index: int, start=False):
        self.index = index
        self.start = start
        self.transitions = []

    def add_transition(self, new_state: State, content=None, pos_tag=None):
        self.transitions.append(StateTransition(new_state, content, pos_tag))

    def is_final(self):
        return len(self.transitions) == 0

    def __str__(self) -> str:
        transitions = "\n".join([f'--> {t}' for t in self.transitions])
        other_states = "\n".join([f'{t.new_state}' for t in self.transitions])
        return f"State({self.index})\n{transitions}\n{other_states}"

class StateTransition:
    def __init__(self, new_state: State, content=None, post_tag=None):
        self.content = content
        self.post_tag = post_tag
        self.new_state = new_state

    def __str__(self) -> str:
        return f'({self.new_state.index})[content: {self.content}, tag: {self.post_tag}]'

    def can_lemme_transit(self, lemme: Tuple[str, str]) -> bool:
        can_do_it = True
        if self.content:
            if lemme[0] == self.content: # Est-ce bien suffisant ?
                can_do_it = True
            else:
                can_do_it = False
        if self.post_tag:
            # TODO check if the types for verbs and adjectives could be separated differently
            post_tag = lemme[1]
            if self.post_tag == 'VSTEM':
                can_do_it = can_do_it and post_tag[0] == 'V'
            elif self.post_tag == 'ADJSTEM':
                can_do_it = can_do_it and post_tag[0] == 'V'
            elif self.post_tag == 'NSTEM':
                can_do_it = can_do_it and post_tag[0] == 'N'
            elif self.post_tag == 'WORDS' or self.post_tag == 'WORD':
                can_do_it = can_do_it and POS_TAG_NOT_WORD.index(post_tag[0]) < 0
        return can_do_it


class SearchPattern:

    def __init__(self, pattern, pos_tagger):
        """
        The idea here is to :
        1/ stemize the pattern while preserving its keywords
        2/ build a state-transition machine from the different
        :param pattern: The pattern to look for which is a combination
         of valid korean words and keywords (see above)
        :param pos_tagger: The position tagger instance
        """
        self.__raw_pattern = pattern
        my_pattern = pattern
        for keyword, replacement in KEYWORDS_AND_REPLACEMENT:
            my_pattern = my_pattern.replace(keyword, replacement)
        # List tags
        tags = pos_tagger.pos(my_pattern)
        # Remove punctuation and none korean words
        self.__fix_words = [word for (word, word_type) in tags if word_type[0] != 'S']
        #print(my_pattern)
        #print(tags)
        #[('여기', 'NP'), ('는', 'JX'), ('VSTEMMMMMMMMMMMM', 'SL'), ('STARTCONDITIONAL', 'SL'), ('교실', 'NNP'), ('ENDCONDITIONAL', 'SL'), ('이', 'VCP'), ('ㅂ니다', 'EC')]

        start = State(-1, start=True)

        fork_states = []
        states_to_link = [start]
        index = 0
        for content, pos_tag in tags:
            if pos_tag == 'SL': # Keywords
                keyword = get_kw_from_replacement(content)
                pos_tag = None
                if keyword == '<VSTEM>':
                    pos_tag = 'VV'
                elif keyword == '<ADJSTEM>':
                    pos_tag = 'VA'
                elif keyword == '<NSTEM>':
                    pos_tag = 'NN'
                elif keyword == '(':
                    fork_states.extend(states_to_link)
                elif keyword == ')':
                    states_to_link.append(fork_states.pop())
                if pos_tag:
                    new_state = State(index)
                    index += 1
                    for state in states_to_link:
                        state.add_transition(new_state=new_state, pos_tag='NN')
                    states_to_link = [new_state]
            else: # Content
                new_state = State(index)
                index += 1
                for state in states_to_link:
                    state.add_transition(new_state=new_state, content=content, pos_tag=pos_tag)
                states_to_link = [new_state]
        self.state_machine = start
        print(self.state_machine)


    def get_fix_words(self) -> [str]:
        return self.__fix_words


komoran = Komoran()

#print(komoran.pos("여기는 VSTEM 교실입니다"))

SearchPattern("여기는 <VSTEM> (교실)입니다", komoran)