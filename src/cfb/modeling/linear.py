"""Regularized linear learner: ridge on a design matrix fit on training rows only.

Design (LinearDesign):
  - rating: missing values filled with the training median of the same star
    band (transfers with stars but no rating), else the overall median; unrated
    rows are carried by the `unrated` flag.
  - other numeric columns: training median + a <col>_na flag when any is missing.
  - booleans as 0/1; categoricals one-hot (unseen levels -> all zeros).
  - rating x position interactions, so the market price can have a different
    slope at each position (the baseline gets this too, for fairness).
  - everything standardized inside the ridge pipeline.
Alpha is chosen by GroupKFold CV on person_id inside the training years.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cfb.modeling.data import CATEGORICAL

ALPHAS = np.logspace(-2, 4, 13)
CV_FOLDS = 5


class LinearDesign:
    def __init__(self, features):
        self.features = list(features)

    def fit(self, df):
        f = self.features
        self.categorical = [c for c in f if c in CATEGORICAL]
        self.boolean = [c for c in f if c not in self.categorical and df[c].dtype == bool]
        self.numeric = [c for c in f if c not in self.categorical + self.boolean]
        self.medians = {c: df[c].median() for c in self.numeric}
        self.flag = [c for c in self.numeric if c != "rating" and df[c].isna().any()]
        if "rating" in f:
            stars = df["stars"] if "stars" in df else pd.Series(np.nan, index=df.index)
            self.band_median = df.groupby(stars).rating.median().to_dict()
        self.levels = {c: sorted(df[c].dropna().unique()) for c in self.categorical}
        return self

    def transform(self, df):
        out = {}
        for c in self.numeric:
            x = df[c].astype(float)
            if c == "rating" and "stars" in df:
                x = x.fillna(df["stars"].map(self.band_median))
            out[c] = x.fillna(self.medians[c] if pd.notna(self.medians[c]) else 0.0)
        for c in self.flag:
            out[f"{c}_na"] = df[c].isna().astype(float)
        for c in self.boolean:
            out[c] = df[c].astype(float)
        for c in self.categorical:
            for level in self.levels[c]:
                out[f"{c}={level}"] = df[c].eq(level).fillna(False).astype(float)
        if "rating" in self.numeric and "position" in self.categorical:
            for level in self.levels["position"]:
                out[f"rating:position={level}"] = out["rating"] * out[f"position={level}"]
        return pd.DataFrame(out, index=df.index)


class RidgeModel:
    name = "ridge"

    def __init__(self, features, target="par_window"):
        self.features, self.target = list(features), target

    def fit(self, train):
        self.design = LinearDesign(self.features).fit(train)
        X, y = self.design.transform(train), train[self.target].to_numpy(float)
        search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                              {"ridge__alpha": ALPHAS}, scoring="neg_mean_squared_error",
                              cv=GroupKFold(n_splits=CV_FOLDS))
        search.fit(X, y, groups=train.person_id.to_numpy())
        self.pipeline, self.alpha = search.best_estimator_, search.best_params_["ridge__alpha"]
        self.columns = list(X.columns)
        return self

    def predict(self, df):
        return self.pipeline.predict(self.design.transform(df))

    def coefficients(self) -> pd.Series:
        """Standardized coefficients (per 1 SD of each design column)."""
        return pd.Series(self.pipeline[-1].coef_, index=self.columns).sort_values(
            key=np.abs, ascending=False)
