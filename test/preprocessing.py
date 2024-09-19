import re
from itertools import chain
from typing import Collection

import paths
from cuentos import Cuento
from preprocess import FileCustomTokenizer, FileCustomTokenizerMaj


class A(Collection):
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __contains__(self, __x: object) -> bool:
        return self.items.__contains__(__x)


def main(cuento: Cuento):
    p = FileCustomTokenizer()
    a = A([cuento.read()])
    p.process(a)
    cls = [i for i in list(a.items[0])[1:-1] if re.match(r'\w+', i)]
    pls = list(cuento.probs()['originales'])
    for c, p in zip(cls, pls):
        print(c, p)


def main_maj(cuento: Cuento):
    p = FileCustomTokenizerMaj()
    a = A([cuento.read()])
    p.process(a)
    for w in a.items[0]:
        print(w)


def count_words(it):
    p = FileCustomTokenizerMaj()
    a = A(list(it))
    p.process(a)
    print()
    print("cantidad de tokens", len(p.dd))
    print(1)


if __name__ == '__main__':
    # main_maj(abierta)
    # count_words(it=(paths.wiki / 'docs').iterdir())
    # count_words(it=paths.ft_prep.iterdir())
    count_words(it=chain(paths.ft_prep.iterdir(), (paths.wiki / 'docs').iterdir()))
