import time

import torch.cuda
from fastai.text import TextList, NumericalizeProcessor, language_model_learner, AWD_LSTM

import paths
from preprocess import FileCustomTokenizer

if torch.cuda.is_available():
    torch.cuda.set_device('cuda:1')
    print("running on cuda")


def main(bs, pretrained_folder):
    (paths.models / 'ft').mkdir(exist_ok=True, parents=True)
    t = time.time()
    # noinspection PyTypeChecker
    data_lm = (
        TextList.from_folder(paths.ft_prep, processor=[FileCustomTokenizer(), NumericalizeProcessor()])
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()
            .databunch(bs=bs, num_workers=1)
    )
    print("databunch", (time.time() - t) / 60)
    t = time.time()

    learn_lm = language_model_learner(
        data_lm,
        AWD_LSTM,
        pretrained_fnames=(pretrained_folder / 'model', pretrained_folder / 'model_vocab'),
        drop_mult=1.0
    ).to_fp16()
    print("load_lm", (time.time() - t) / 60)

    lr = 1e-3
    lr *= bs / 48

    dict(learn_lm.model.named_modules())['1.decoder'].reset_parameters()

    learn_lm.fit_one_cycle(2, lr * 10, moms=(0.8, 0.7))

    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(8, lr, moms=(0.8, 0.7))

    learn_lm.to_fp32()
    learn_lm.save(paths.models / 'ft' / 'model')
    learn_lm.data.vocab.save(paths.models / 'ft' / 'model_vocab.pkl')
    learn_lm.export(file=paths.models / 'ft' / f'export.pkl')


if __name__ == '__main__':
    main(
        bs=128,
        pretrained_folder=paths.models / '10_all'
    )
