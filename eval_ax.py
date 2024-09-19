from typing import List

import pandas as pd

import cuentos
from cuentos import Cuento


def main(cuentos: List[Cuento]):
    for c in cuentos:
        print("=" * 80)
        print("reading", c.name)

        model_df = pd.read_csv(c.results())
        prev_df = c.probs()

        print(f"model_df: {model_df.shape}", f'prev_df: {prev_df.shape}')

        df = model_df.join(prev_df, how='inner')

        # print(df)
        print(f"df: {df.shape}")
        print(df.corr())
        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print(df)


if __name__ == '__main__':
    main(cuentos.todos)
