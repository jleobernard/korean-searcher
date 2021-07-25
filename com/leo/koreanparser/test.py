from com.leo.koreanparser.main import find_pattern_in_local_db

corpus = [
    "여기는 교실입니다.",
    "여기는 교실입니다.",
    "회사를 다닌 지 한 달이 됐어요."
]

user_query = "<VSTEM>(으)ㄴ 지 <WORDS> 되다"

found = find_pattern_in_local_db(pattern="여기", db=corpus)
assert len(found) == 1
assert found[0] == corpus[0]

found = find_pattern_in_local_db(pattern="여기<WORDS>이다", db=corpus)
assert len(found) == 2
assert found[0] == corpus[0]
assert found[1] == corpus[1]

found = find_pattern_in_local_db(pattern=user_query, db=corpus)
assert len(found) == 1
assert found[0] == corpus[2]
