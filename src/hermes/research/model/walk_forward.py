"""Walk-forward LightGBM cross-sectional return model.

Strict no-look-ahead: to predict month t's cross-section, the model trains ONLY on
samples whose forward-return target realized strictly before t (i.e. signal dates
< t), inside a rolling window, retrained every month. Output is an OUT-OF-SAMPLE
signal panel (date x code) usable by signal_portfolio_backtest / factor eval.

Target is the cross-sectionally DEMEANED forward return (the market component is
removed, so the model learns relative ranking, not beta).
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from ..factors import library as fl

DEFAULT_PARAMS = dict(
    n_estimators=200, num_leaves=15, learning_rate=0.03, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1,
)


def build_dataset(factors: dict[str, pd.DataFrame], close: pd.DataFrame, eval_dates: list,
                  members_asof=None) -> tuple[pd.DataFrame, list[str]]:
    """Long table of (date, code) rows at `eval_dates`: standardized factor features +
    `fwd_ret` (cross-sectionally demeaned next-period return). Returns (data, feature_cols).

    PIT-correct: features are restricted to the then-current members BEFORE the
    cross-sectional standardize() -- standardizing over the survivorship-defined union
    would leak future membership into the z-scores (see factors.library.restrict_to_universe).
    Aligned to the monthly `eval_dates` only (the rest of the panel is never used here)."""
    aligned = {k: v.reindex(index=eval_dates, columns=close.columns) for k, v in factors.items()}
    if members_asof is not None:
        aligned = {k: fl.restrict_to_universe(v, members_asof) for k, v in aligned.items()}
    feats = {k: fl.standardize(v) for k, v in aligned.items()}
    cols = list(feats)
    blocks = []
    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        fwd = close.loc[t1] / close.loc[t] - 1.0
        block = pd.DataFrame({k: feats[k].loc[t] for k in cols})
        block["fwd_ret"] = fwd
        if members_asof is not None:
            block = block.loc[block.index.isin(members_asof(t))]
        block = block.dropna(subset=cols + ["fwd_ret"])
        if block.empty:
            continue
        block["fwd_ret"] -= block["fwd_ret"].mean()    # cross-sectional demean (remove beta)
        block["date"] = t
        block["code"] = block.index
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True), cols


def walk_forward_predict(data: pd.DataFrame, feature_cols: list[str], min_train: int = 24,
                         window: int = 36, params: dict | None = None) -> pd.DataFrame:
    """Rolling-window walk-forward predictions -> OOS signal panel (date x code).

    For the k-th signal date (k >= min_train), train on the prior `window` dates
    (strictly earlier) and predict that date's cross-section."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    udates = sorted(data["date"].unique())
    preds = {}
    for idx, t in enumerate(udates):
        if idx < min_train:
            continue
        train_dates = udates[max(0, idx - window):idx]     # strictly before t
        tr = data[data["date"].isin(train_dates)]
        te = data[data["date"] == t]
        if len(tr) < 200 or te.empty:
            continue
        model = LGBMRegressor(**p)
        model.fit(tr[feature_cols], tr["fwd_ret"])
        preds[t] = pd.Series(model.predict(te[feature_cols]), index=te["code"].to_numpy())
    signal = pd.DataFrame(preds).T
    signal.index = pd.to_datetime(signal.index)
    return signal.sort_index()
