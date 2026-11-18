"""LightGBM learner (squared error), tuned by grouped CV inside the training years.

Missing values are left to LightGBM. Categorical columns use the categories
seen in training (unseen levels -> missing). The grid is small on purpose:
training sets are ~1,300-1,600 rows. For each grid point lgb.cv runs
GroupKFold on person_id with early stopping; the best point's round count is
used to refit on all training rows.
"""

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cfb.modeling.data import CATEGORICAL

GRID = {"num_leaves": [4, 8, 16], "min_data_in_leaf": [20, 50, 100], "lambda_l2": [0.0, 10.0]}
BASE_PARAMS = {"objective": "regression", "learning_rate": 0.03, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 0, "deterministic": True,
               "num_threads": 1, "verbosity": -1}
MAX_ROUNDS = 2000
EARLY_STOP = 100
CV_FOLDS = 5


class LGBMModel:
    name = "lgbm"

    def __init__(self, features, target="par_window"):
        self.features, self.target = list(features), target

    def _frame(self, df):
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            if c in self.categories:
                X[c] = pd.Categorical(df[c].astype(object), categories=self.categories[c])
            elif pd.api.types.is_bool_dtype(df[c]):
                X[c] = df[c].astype(float)
            else:
                X[c] = df[c].astype(float)
        return X

    def fit(self, train):
        self.categories = {c: sorted(train[c].dropna().unique()) for c in self.features
                           if c in CATEGORICAL}
        X, y = self._frame(train), train[self.target].to_numpy(float)
        folds = list(GroupKFold(n_splits=CV_FOLDS).split(X, y, groups=train.person_id))
        best = None
        for values in itertools.product(*GRID.values()):
            params = {**BASE_PARAMS, **dict(zip(GRID, values))}
            data = lgb.Dataset(X, y, categorical_feature=list(self.categories) or "auto",
                               free_raw_data=False)
            res = lgb.cv(params, data, num_boost_round=MAX_ROUNDS, folds=folds,
                         callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            key = next(k for k in res if k.endswith("l2-mean"))
            score, rounds = min(res[key]), int(np.argmin(res[key])) + 1
            if best is None or score < best[0]:
                best = (score, rounds, params)
        self.cv_l2, self.rounds, self.params = best
        self.booster = lgb.train(self.params, lgb.Dataset(
            X, y, categorical_feature=list(self.categories) or "auto"),
            num_boost_round=self.rounds)
        return self

    def predict(self, df):
        return self.booster.predict(self._frame(df))

    def shap_values(self, df) -> pd.DataFrame:
        """Exact TreeSHAP contributions (LightGBM pred_contrib); the last column
        is the expected value, and each row sums to the prediction."""
        contrib = self.booster.predict(self._frame(df), pred_contrib=True)
        return pd.DataFrame(contrib, index=df.index, columns=self.features + ["expected_value"])

    def describe(self) -> str:
        p = {k: self.params[k] for k in GRID}
        return f"rounds {self.rounds}, {p}, cv l2 {self.cv_l2:.1f}"
