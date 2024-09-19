from pathlib import Path

root: Path = Path(__file__).parent
data = root / 'data'
fastai = data / 'fastai'
wiki = data / 'wiki'
epubs = data / 'epubs'
models = data / 'models'
cuentos = root / 'cuentos_test'
ft_train = data / 'spanish_books_full'
ft_prep = data / 'spanish_books_full_preprocessed'
resultados = root / 'resultados'
plots = root / 'plots'


for path in [root, data, fastai, wiki, models, ft_prep, plots]:
    if not path.exists():
        path.mkdir()

