import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import current_models as cm
import paths

plt.rcParams['svg.fonttype'] = 'none'


def aic_all():
    ft = aic_df(cm.fine_tuned).loc['aic', :]
    df = aic_df(cm.default).loc['aic', :]
    df_maj = aic_df(cm.default_maj).loc['aic', :]
    ft_maj = aic_df(cm.fine_tuned_maj).loc['aic', :]
    df = pd.DataFrame({'ft': ft, 'def': df, 'ft_maj': ft_maj, 'def_maj': df_maj})
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.xaxis.tick_top()
    df.loc[[f"M{i}" for i in range(9, 17)]].plot.bar(rot=0, ax=ax)
    plt.savefig(paths.plots / f'aic.svg', transparent=True)


def aic_all_2():
    df_maj = aic_df(cm.default_maj).loc[['aic'], :]
    ft_maj = aic_df(cm.fine_tuned_maj).loc[['aic'], 'M9':]
    ft_maj.columns = ['F'+c for c in ft_maj.columns]
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.xaxis.tick_top()
    df_maj.join(ft_maj).T.plot.bar(rot=0, ax=ax)
    plt.savefig(paths.plots / f'aic2.svg', transparent=True)


def aic_df(prefix):
    df = pd.read_csv(paths.resultados / prefix / f'aic.csv', index_col=0)
    return (df - df.loc['aic', 'M0']).drop('M0', axis=1)


def t_values_plot(prefix):
    plt.figure(figsize=(8, 7))
    df = get_df_t_value_plot(prefix)
    df = df[['M0']].join(df.loc[:, 'M9':])
    font = {
        'family': 'Raleway',
        'weight': 'bold'
    }
    sns.heatmap(df, annot=True, center=0, cmap='coolwarm', vmin=-20, vmax=20, fmt="0.2f", annot_kws=font)
    (paths.plots / prefix).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(paths.plots / prefix / f't_values.svg', transparent=True)
    plt.savefig(paths.plots / prefix / f't_values.jpg')


def t_value_plot_both():
    df_def = get_df_t_value_plot(cm.default_maj)
    df_ft = get_df_t_value_plot(cm.fine_tuned_maj).loc[:, 'M9':]
    df_ft.columns = ['F' + c for c in df_ft.columns]
    df = df_def.join(df_ft)
    # df.drop(index=['rpt', 'rps', 'rpl'], inplace=True)
    plt.figure(figsize=(20, 3.5))
    font = {'family': 'Raleway', 'weight': 'bold', 'size': 9}
    sns.heatmap(df, annot=True, center=0, cmap='bwr', vmin=-20, vmax=20, fmt="0.2f", annot_kws=font)
    plt.tight_layout()
    plt.savefig(paths.plots / f't_values_all.svg', transparent=True)


def get_df_t_value_plot(prefix):
    df = pd.read_csv(paths.resultados / prefix / 'tvalues.csv', index_col=0).drop('(Intercept)', axis=0)
    df = pd.concat([df.drop(['CLOZE_pred_remef']), pd.DataFrame(df.loc['CLOZE_pred_remef', :]).T])
    df.rename(inplace=True, index={
        'X4.gramcache.0.0001500000_0.15': '4-gram+cache',
        'LSA009.promedio.conSW': 'CS-LSA (w=9)',
        'FT050.distancia.promedio_conSW_wiki': 'CS-FT (w=50)'
    })
    return df


def aic_plot(prefix):
    df = aic_df(prefix).loc[:, 'M9':]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.xaxis.tick_top()
    df.T.plot.bar(rot=0, ax=ax)
    (paths.plots / prefix).mkdir(exist_ok=True)
    plt.savefig(paths.plots / prefix / f'aic.svg', transparent=True)


def main(prefix):
    aic_plot(prefix)
    t_values_plot(prefix)


if __name__ == '__main__':
    t_value_plot_both()
    main(cm.default)
    main(cm.fine_tuned)
    main(cm.default_maj)
    main(cm.fine_tuned_maj)
    # aic_all()
    # aic_all_2()
