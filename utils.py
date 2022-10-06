from scipy.stats import probplot, moment
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
from statsmodels.tsa.stattools import adfuller, q_stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0], title='Time Series')
    x.rolling(14).mean().plot(ax=axes[0][0], c='k', lw=1)
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1], dist="norm")
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


def combine_data(dict_of_df: Dict) -> pd.DataFrame:
    df = pd.DataFrame()
    for _, data in dict_of_df.items():
        df = pd.concat([df, data])
    return df.sort_index()


def unpack_hierarchical_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    indexes = df.columns.get_level_values(0).unique()
    df_unpack = pd.DataFrame()
    for i in indexes:
        temp_df = df[i].copy(deep=True)
        temp_df["home_num"] = i
        df_unpack = pd.concat([df_unpack, temp_df])
    return df_unpack


def nmae(pred, act):
    return np.sum(np.abs(pred - act)) / np.sum(act)

def mape(pred, act):
    return np.mean(np.abs(pred-act)/act)

def nmae_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return nmae(pred, y)

def mape_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return mape(pred, y)

def mean_squared_error_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return mean_squared_error(pred, y)

def corr_metric(pred, act):
    return spearmanr(pred, act)[0]

def corr_metric_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return spearmanr(y, pred)[0]


