from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
import pandas as pd
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

"""
All Classes are designed to work on univariate timeseries
"""


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.feature_name]


class CastType(BaseEstimator, TransformerMixin):
    def __init__(self, to_dtype=float):
        self.to_dtype = to_dtype

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.astype(self.to_dtype)


class ApplyLogTransformation(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return np.log(X + 0.01)


class ApplyRollingWindowMean(BaseEstimator, TransformerMixin):
    def __init__(self, widnow: int):
        self.widnow = widnow

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.rolling(self.widnow).mean().to_frame(f"rolling_mean_{self.widnow}_{X.name}")


class ApplyRollingWindowStd(BaseEstimator, TransformerMixin):
    def __init__(self, widnow: int):
        self.widnow = widnow

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.rolling(self.widnow).std().to_frame(f"rolling_std_{self.widnow}_{X.name}")


class CalcShift(BaseEstimator, TransformerMixin):
    def __init__(self, shift_val, new_feat_name=None):
        self.shift_val = shift_val
        self.new_feat_name = new_feat_name

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        if self.new_feat_name is None:
            if self.shift_val <= 0:
                shift_direction = "forward"
            elif self.shift_val > 0:
                shift_direction = "backward"
            self.new_feat_name = f"{shift_direction}_{x.name}_{abs(self.shift_val)}"
        return x.shift(self.shift_val).to_frame(self.new_feat_name)


class ExtractDayTime(BaseEstimator, TransformerMixin):
    # constructed by visual inspection of boxplot
    # consumption during day time
    # mapper = {
    #     0 : "Night",
    #     1 : "Night",
    #     2 : "Night",
    #     3 : "Night",
    #     4 : "Night",
    #     5 : "Morning",
    #     6 : "Morning",
    #     7 : "Morning",
    #     8 : "Morning",
    #     9 : "Day Time",
    #     10 : "Day Time",
    #     11 : "Day Time",
    #     12 : "Day Time",
    #     13 : "Day Time",
    #     14 : "Day Time",
    #     15 : "Day Time Peak Hour",
    #     16 : "Day Time Peak Hour",
    #     17 : "Day Time Peak Hour",
    #     18 : "Day Time Peak Hour",
    #     19 : "Day Time Peak Hour",
    #     20 : "Day Time Peak Hour",
    #     21 : "Evening",
    #     22 : "Evening",
    #     23 : "Evening"
    # }
    mapper = {
        0: "Low Consumption Hour",
        1: "Low Consumption Hour",
        2: "Low Consumption Hour",
        3: "Low Consumption Hour",
        4: "Low Consumption Hour",
        5: "Low Consumption Hour",
        6: "Low Consumption Hour",
        7: "Low Consumption Hour",
        8: "Low Consumption Hour",
        9: "Low Consumption Hour",
        10: "Low Consumption Hour",
        11: "Low Consumption Hour",
        12: "Low Consumption Hour",
        13: "Low Consumption Hour",
        14: "Low Consumption Hour",
        15: "High Consumption Hour",
        16: "High Consumption Hour",
        17: "High Consumption Hour",
        18: "High Consumption Hour",
        19: "High Consumption Hour",
        20: "High Consumption Hour",
        21: "High Consumption Hour",
        22: "Low Consumption Hour",
        23: "Low Consumption Hour"
    }
    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        return x.apply(lambda val: ExtractDayTime.mapper[val.hour]).to_frame("day_time")


class ExtractWeekDay(BaseEstimator, TransformerMixin):

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        mapper = {0: "workday",
                  1: "workday",
                  2: "workday",
                  3: "workday",
                  4: "workday",
                  5: "weekend",
                  6: "weekend"}

        return x.apply(lambda val: mapper[val.weekday()]).to_frame("week_day")


class ExtractSeason(BaseEstimator, TransformerMixin):

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        seasons = {
            1: 'Winter',
            2: 'Winter',
            3: 'Spring',
            4: 'Spring',
            5: 'Spring',
            6: 'Summer',
            7: 'Summer',
            8: 'Summer',
            9: 'Autumn',
            10: 'Autumn',
            11: 'Autumn',
            12: 'Winter'
        }
        return x.apply(lambda val: seasons[val.month]).to_frame("season")


class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self, drop_first=False):
        self.drop_first = drop_first

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        return pd.get_dummies(x.astype(str), drop_first=self.drop_first)


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs.dropna()

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)


    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs.dropna()


def make_union(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return PandasFeatureUnion(
        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)

