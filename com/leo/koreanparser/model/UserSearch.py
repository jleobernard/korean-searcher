import jamotools
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Komoran

"""
Idée de l'algo :
T <- Récupérer de la base la liste des phrases contenant tous les mots fixes.
resultats <- []
POUR CHAQUE t de T FAIRE
    mmElts <- Multimap (élément de la requête; liste des positions de l'élément de la requête) 
    SI un chemin existe satisfiant la requête ALORS
        resultats <- resultats + t
    FSI
FIN POUR
"""


class UserSearch:

    __matchers: [str]

    def __init__(self, query: str):
        """
        :param query: The query made by the user. It should be contain the following elements:
            - <word>         : any series of characters
            - VSTEM          : keyword that represents a verb stem
            - ADJSTEM        : keyword that represents an adjective stem
            - NSTEM          : keyword that represents a noun stem
            - NOUN           : a noun
            - (<word>)       : maybe the <word>
            - <word>/<word>  : one of this word
            - -<word>        : word attached to another word
        """
        pass

def split(word):
    return [char for char in word]

haystack = "딩가딩가할래"
splitted = split(haystack)
print(f"{jamotools.split_syllable_char(splitted[0])}")

#okt = Okt()
#print(okt.pos(u'사촌 결혼 덕분에 뉴욕에 다녀왔어요.'))

result=[('사촌', 'Noun'), ('결혼', 'Noun'), ('덕분', 'Noun'), ('에', 'Josa'), ('뉴욕', 'Noun'), ('에', 'Josa'), ('다녀왔어요', 'Verb'), ('.', 'Punctuation')]


#okt = Okt()
#print(okt.pos(u'그것은 차입니까'))
kkma = Kkma()
#print(kkma.pos(u'생길지도'))
#[('생기', 'VV'), ('ㄹ지', 'ECD'), ('도', 'JX')]
#print(kkma.pos(u'몰라요'))
# [('몰', 'VV'), ('ㄹ라요', 'EFN')]
#print(kkma.pos(u'모르다'))
print(kkma.pos(u'이제 자야겠어요'))

komoran = Komoran()
"""
print(komoran.pos(u'사촌 결혼 덕분에 뉴욕에 다녀왔어요'))
print(komoran.pos(u'생길지도'))
print(komoran.pos(u'몰라요'))
print(komoran.pos(u'모르다'))
print(komoran.pos(u'을지도 모르다'))
print(komoran.pos(u'그것은 차입니까'))
"""
print(komoran.pos(u'이제 자야겠어요'))