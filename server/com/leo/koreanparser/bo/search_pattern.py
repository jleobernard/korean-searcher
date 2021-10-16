from __future__ import annotations
from typing import Tuple

Lemme = Tuple[str, str]
Sentence = [Lemme]

def get_kw_and_replacement(keyword: str):
    if keyword == '(':
        replacement = "STARTCONDITIONAL"
    elif keyword == ')':
        replacement = "ENDCONDITIONAL"
    else:
        replacement = keyword[1:-1] + "MMMM"
    return keyword, replacement


def get_kw_from_replacement(replacement: str):
    replacements = [kw for kw, rpl in KEYWORDS_AND_REPLACEMENT if rpl == replacement]
    return replacements[0] if len(replacements) > 0 else None


KEYWORDS = ['<VSTEM>', '<ADJSTEM>', '≤NSTEM>', '<WORDS>', '<WORD>', '<VENDING>', '(', ')']
KEYWORDS_AND_REPLACEMENT = [get_kw_and_replacement(kw) for kw in KEYWORDS]
POS_TAG_NOT_WORD = ['SC', 'SE', 'SF', 'SSC', 'SSO']


class State:
    def __init__(self, index: int, start=False):
        self.index = index
        self.start = start
        self.transitions: [StateTransition] = []

    def add_transition(self, new_state: State, content=None, pos_tag=None):
        self.transitions.append(StateTransition(new_state, content, pos_tag))

    def is_final(self):
        return len(self.transitions) == 0

    def __str__(self, already_printed=[]) -> str:
        already_printed.append(self.index)
        transitions = "\n".join([f'--> {t}' for t in self.transitions])
        other_states = "\n".join([f'{t.new_state.__str__(already_printed)}' for t in self.transitions if t.new_state.index not in already_printed])
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
            elif self.post_tag == 'VENDING':
                can_do_it = can_do_it and post_tag[0] == 'E'
            elif self.post_tag == 'WORDS' or self.post_tag == 'WORD':
                can_do_it = can_do_it and post_tag[0] not in POS_TAG_NOT_WORD
        return can_do_it


class SearchPattern:

    def __init__(self, pattern, pos_tagger):
        """
        The idea here is to :
        1/ stemize the pattern while preserving its keywords and replacing infinitive
        form of verbs by a pattern matching the conjugated verb
        2/ build a state-transition machine from the different
        :param pattern: The pattern to look for which is a combination
         of valid korean words and keywords (see above).
        !!!! Warning !!! <WORDS> must not be a possible start of the pattern
        :param pos_tagger: The position tagger instance
        """
        self.__raw_pattern = pattern
        my_pattern = pattern
        for keyword, replacement in KEYWORDS_AND_REPLACEMENT:
            my_pattern = my_pattern.replace(keyword, replacement)
        # List tags
        tags = pos_tagger.pos(my_pattern)
        # TODO trouver une parade pour 이다
        for i in range(0, len(tags) - 1):
            tag = tags[i]
            if tag[1][0] == 'V' and tags[i + 1][0] == '다':
                tags[i + 1] = (get_kw_and_replacement('<VENDING>')[1], 'SL')

        # Remove punctuation and none korean words
        self.__fix_words = [lemme for lemme in tags if lemme[0] != 'S']

        start = State(-1, start=True)

        fork_states = []
        states_to_link = [start]
        index = 0
        for content, pos_tag in tags:
            if pos_tag == 'SL': # Keywords
                keyword = get_kw_from_replacement(content)
                pos_tag_to_match = None
                if keyword == '<VSTEM>':
                    pos_tag_to_match = 'VSTEM'
                elif keyword == '<ADJSTEM>':
                    pos_tag_to_match = 'ADJSTEM'
                elif keyword == '<NSTEM>':
                    pos_tag_to_match = 'NSTEM'
                elif keyword == '<VENDING>':
                    for state in states_to_link:
                        state.add_transition(new_state=state, pos_tag='VENDING')
                elif keyword == '<WORDS>':
                    for state in states_to_link:
                        state.add_transition(new_state=state, pos_tag='WORDS')
                elif keyword == '(':
                    fork_states.extend(states_to_link)
                elif keyword == ')':
                    states_to_link.append(fork_states.pop())
                if pos_tag_to_match:
                    new_state = State(index)
                    index += 1
                    for state in states_to_link:
                        state.add_transition(new_state=new_state, pos_tag=pos_tag_to_match)
                    states_to_link = [new_state]
            else: # Content
                new_state = State(index)
                index += 1
                for state in states_to_link:
                    state.add_transition(new_state=new_state, content=content, pos_tag=pos_tag)
                states_to_link = [new_state]
        self.state_machine = start

    def get_fix_words(self) -> [Lemme]:
        return self.__fix_words

    def matches(self, haystack: [Tuple[str, str]], start=0) -> bool:
        curr_states = [self.state_machine]
        for i in range(start, len(haystack)):
            lemme = haystack[i]
            new_states = {}
            for curr_state in curr_states:
                for transition in curr_state.transitions:
                    new_state = transition.new_state
                    if new_state.index not in new_states and transition.can_lemme_transit(lemme):
                        if new_state.is_final():
                            return True
                        new_states[new_state.index] = new_state
            curr_states = new_states.values()
            if len(curr_states) == 0:
                return False
        return False

    def get_starts(self) -> [StateTransition]:
        return self.state_machine.transitions
