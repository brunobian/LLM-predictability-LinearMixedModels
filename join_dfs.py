import pandas as pd

import cuentos
import paths
from cuentos import Cuento
import scipy.special

import current_models as cm


def as_rdf(cuento: Cuento, prefix):
    df = pd.read_csv(cuento.results(prefix),index_col=False)
    df['rpt_abs'] = df.index + 1
    df['textid'] = cuento.r_index
    df['awd'] = scipy.special.logit(df['prob']).clip(-20)
    df['awd_prob'] = df['prob']
    df['awd_word'] = df['word']
    df['awd_cuento'] = cuento.name[2:]
    return df[['rpt_abs', 'textid', 'awd', 'awd_prob', 'awd_word', 'awd_cuento']]


def main(prefix=''):
    df = pd.concat(as_rdf(c, prefix) for c in cuentos.todos)
    (paths.data / '..' / 'resultados' / prefix).mkdir(exist_ok=True)
    df.to_csv(paths.data / '..' / 'resultados' / prefix / f'all.csv', index=False)


if __name__ == '__main__':
    #main(cm.default)
    #main(cm.fine_tuned)
    #main(cm.default_maj)
    #main(cm.fine_tuned_maj)
    #main('lstm')
    #main('clm-spanish')
    #main(cm.gpt2_ft)
    #main(cm.gpt2_blogs)
    #main(cm.llama)
    main(cm.llama2)

