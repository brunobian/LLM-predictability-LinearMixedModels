from typing import NamedTuple

import pandas as pd

import paths


class Cuento(NamedTuple):
    name: str
    index: int
    r_index: int

    def read(self):
        with open(paths.cuentos / f'{self.name}.txt') as f:
            return f.read()

    def probs(self):
        df = pd.read_csv(paths.cuentos / 'predictability.csv', usecols=["tId", "misWords", "originales", "pred"])
        df = df[df.tId == self.index]
        df.misWords -= 1
        df = df.set_index("misWords")
        return df[["originales", "pred"]]

    def results(self, mn=""):
        (paths.data / 'cuentos' / mn).mkdir(exist_ok=True)
        return paths.data / 'cuentos' / mn / f'{self.name}.csv'


abierta = Cuento(name='1 Carta abierta', index=7, r_index=1)
bob = Cuento(name='2 Bienvenido Bob', index=1, r_index=2)
axolotl = Cuento(name='3 axolotl', index=0, r_index=3)
especies = Cuento(name='5 El origen de las especies', index=5, r_index=6)
sacks = Cuento(name='6 sacks - rebeca', index=2, r_index=7)
loco = Cuento(name='7 el loco cansino', index=3, r_index=8)
negro = Cuento(name='8 el negro de paris', index=4, r_index=9)
senorita = Cuento(name='9 carta a una senorita en paris', index=6, r_index=10)

todos = [abierta, bob, axolotl, especies, sacks, loco, negro, senorita]
