from fastai.text import load_learner, LanguageLearner, warn, torch, UNK

import paths


def load_ft_model():
    pass


def predict(learner, text: str, n_words: int = 1, temperature: float = 1., min_p: float = None):
    """Return `text` and the `n_words` that come after"""
    learner.model.reset()
    xb, yb = learner.data.one_item(text)
    xb = xb[:, :-1]
    new_idx = []
    for _ in range(n_words):  # progress_bar(range(n_words), leave=False):
        res = learner.pred_batch(batch=(xb, yb))[0][-1]
        if min_p is not None:
            if (res >= min_p).float().sum() == 0:
                warn(f"There is no item with probability >= {min_p}, try a lower value.")
            else:
                res[res < min_p] = 0.
        res[learner.data.vocab.stoi[UNK]] = 0.
        if temperature != 1.:
            res.pow_(1 / temperature)
        idx = torch.multinomial(res, 1).item()
        new_idx.append(idx)
        xb = xb.new_tensor([idx])[None]
    return text + ' ' + learner.data.vocab.textify(new_idx, sep=' ')


def main():
    learn: LanguageLearner = load_learner(paths.models / 'ft')
    # learn: LanguageLearner = load_learner(paths.models / '10_all')

    while True:
        text = input("-->")
        if text == 'exit':
            break
        print(predict(learn, text=text, n_words=500, temperature=0.5))


if __name__ == '__main__':
    main()
