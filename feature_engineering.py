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

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        return x.apply(lambda val: val.hour).to_frame("day_time")


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

