"""SHAP summaries for the LightGBM models (exact TreeSHAP from LightGBM).

SHAP values explain what the MODEL does with each feature on the test year;
they are not causal effects, and correlated features (e.g. rating and stars)
share credit in ways that can look arbitrary.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def shap_summary(model, df: pd.DataFrame) -> pd.DataFrame:
    """Per feature: mean |SHAP|, share of the total, and direction.

    direction (numeric / boolean features): Spearman between the feature value
    and its SHAP value, +1 = higher value -> higher prediction. Categorical
    features get mean SHAP per level in `levels` instead.
    """
    shap = model.shap_values(df).drop(columns="expected_value")
    mean_abs = shap.abs().mean()
    rows = []
    for c in shap.columns:
        row = {"feature": c, "mean_abs_shap": mean_abs[c],
               "share": mean_abs[c] / mean_abs.sum(), "direction": np.nan, "levels": ""}
        if c in model.categories:
            by_level = shap[c].groupby(df[c].astype(object).fillna("NA")).mean()
            row["levels"] = ", ".join(f"{k} {v:+.1f}" for k, v in
                                      by_level.sort_values(ascending=False).items())
        else:
            x = df[c].astype(float)
            ok = x.notna()
            if ok.sum() > 2 and x[ok].nunique() > 1 and shap.loc[ok, c].nunique() > 1:
                row["direction"] = spearmanr(x[ok], shap.loc[ok, c]).statistic
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def format_summary(summary: pd.DataFrame) -> str:
    s = summary.copy()
    s["mean_abs_shap"] = s.mean_abs_shap.round(2)
    s["share"] = (100 * s.share).round(1).astype(str) + "%"
    s["direction"] = s.direction.round(2)
    return s.to_string(index=False)
