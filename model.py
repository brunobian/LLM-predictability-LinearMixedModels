import tarfile

import numpy as np
import requests
import torch.cuda
from fastai.basic_data import DataBunch
from fastai.datasets import Config, shutil
from fastai.text import (
    TextList, language_model_learner, AWD_LSTM, load_learner, LanguageLearner,
    NumericalizeProcessor,
)

import cuentos
import paths
from eval_model import evaluate
from preprocess import FileCustomTokenizerMaj

if torch.cuda.is_available():
    torch.cuda.set_device('cuda:1')
    print("running on cuda")


def download_wiki():
    url = "https://docs.google.com/uc?export=download"
    file_id = '1KaBe20MRxQNJ14c-_ZoVQ8z9fAKbNRJ1'
    wiki_file = paths.wiki / 'wiki.tar.bz2'

    if not wiki_file.exists():
        session = requests.Session()
        response = session.get(url, params={'id': file_id}, stream=True)
        token = next(
            (value for key, value in response.cookies.items() if key.startswith('download_warning')),
            None
        )

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        CHUNK_SIZE = 32768
        with wiki_file.open('wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    folder = paths.wiki / 'docs'
    if not folder.exists():
        with tarfile.open(wiki_file, "r:bz2") as tar:
            tar.extractall(paths.wiki)

    return folder


def make_small(size: int):
    rs = np.random.RandomState(size)
    small = paths.data / f'small_{size}'
    if not small.exists():
        small.mkdir(exist_ok=True)
        (small / 'docs').mkdir(exist_ok=True)
        for f in rs.choice((paths.wiki / 'docs').ls(), size=size, replace=False):
            shutil.copy(f, small / 'docs' / f.name)
    return small / 'docs'


def train(folder, bs, size, n):
    model_folder = paths.models / f'{n}_{size}'
    if not (model_folder / 'model_vocab.pkl').exists():
        model_folder.mkdir(exist_ok=True)
        # noinspection PyTypeChecker
        data: DataBunch = (
            TextList
                .from_folder(folder, processor=[FileCustomTokenizerMaj(), NumericalizeProcessor()])
                .split_by_rand_pct(0.1, seed=42)
                .label_for_lm()
                .databunch(bs=bs, num_workers=1)
        )
        data.save(model_folder / 'data.databunch')

        print('Data bunch crated')
        learn: LanguageLearner = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False)
        lr = 2e-2
        learn.unfreeze()
        learn.fit_one_cycle(n, lr / n, moms=(0.8, 0.7))
        learn.to_fp32()
        learn.save(model_folder / 'model')
        learn.data.vocab.save(model_folder / 'model_vocab.pkl')
        learn.export(file=model_folder / f'export.pkl')
    return model_folder


def main():
    bs = 128
    print(Config.data_path())
    download_wiki()
    size = 'all'
    folder = make_small(size) if isinstance(size, int) else paths.wiki / 'docs'
    model_folder = train(folder, bs, size, n=10)
    learn: LanguageLearner = load_learner(model_folder)

    for c in cuentos.todos:

        print(f"predicting {c.name}")

        content = c.read()
        res = evaluate(learn, content)
        with (c.results()).open('w') as f:
            f.write('word,prob\n')
            for word, prob in res:
                f.write(f'{word},{prob}\n')


if __name__ == '__main__':
    train(
        folder=paths.wiki / 'docs',
        bs=128,
        size='all_epubs',
        n=10
    )
    """ main() """