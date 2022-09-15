with open('./wiki.ko.vocab', mode='r', encoding='UTF8') as f:
    content = f.read()
    characters = set()
    for c in content:
        if c not in '0123456789' and (u'\u1100' <= c <= u'\u11FF' or u'\uAC00' <= c <= u'\uD7A3'):
            characters.add(c)
sorted_chars = sorted([c for c in characters])
with open('./wiki.ko.syl', mode='w', encoding='UTF8') as f:
    f.writelines('\n'.join(sorted_chars))