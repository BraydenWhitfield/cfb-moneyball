"""Player-season production: plays, level, replacement level, PPA above replacement.

This is outcome A of the master table. One row per (player_id, season, team)
from the PPA table (offensive skill players, FBS and FCS):

  plays    implied plays, totalPPA / averagePPA for pass and rush, where the
           rounded averages pin it down (|average| >= MIN_ABS_AVERAGE); else
           box-score plays (ATT + CAR + REC) times a per-position calibration
           factor (median implied / box). Box-score counts beat usage share x
           team plays at every position, WR included (see compare_plays_estimators).
  level    FBS / FCS / other (D2, NAIA: partial seasons, valued at 0).
  repl     replacement PPA per play: the REPLACEMENT_QUANTILE of PPA per play
           among regulars (>= MIN_PLAYS) at that position, season, and level.
  par      totalPPA - repl * plays: PPA above a replacement at the same level.
  par_fbs  FBS-equivalent: par at FBS, k * par at FCS, 0 at "other", where k is
           the FCS -> FBS exchange rate estimated from players who moved
           between levels (estimate_exchange_rate).
  role     0 none, 1 rotation (>= MIN_PLAYS), 2 starter (also top STARTERS by
           plays at his position on his team that season). Outcome B.

All player_ids are mapped through id_aliases first, so split players stay whole.
"""

import numpy as np
import pandas as pd

from cfb import clean, load
from cfb.crosswalk import apply_aliases

FBS_CONFERENCES = {"AAC", "ACC", "B12", "B1G", "CUSA", "MAC", "MWC", "PAC", "SBC", "SEC", "Ind"}
MIN_ABS_AVERAGE = 0.05      # averages are rounded to 3 decimals: <= 1% error on plays
MIN_PLAYS = {"QB": 100, "RB": 50, "WR": 20, "TE": 20}
STARTERS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1}
REPLACEMENT_QUANTILE = 0.2
QB_MIN_ATTEMPTS = 25        # season_position: this many passes = a QB that season
RB_MIN_CARRIES = 20
MIN_MOVERS = 50             # fewer level movers than this -> fallback discount
FALLBACK_FCS_DISCOUNT = 0.5


# --- Inputs -------------------------------------------------------------------

def load_inputs(aliases: pd.DataFrame, raw_dir=None):
    """(ppa, stats, usage) with normalized teams and canonical player_ids."""
    kw = {} if raw_dir is None else {"raw_dir": raw_dir}
    ppa = clean.combine_ppa_splits(clean.dedupe_player_seasons(load.load_ppa(**kw)))
    usage = clean.combine_usage_splits(clean.dedupe_player_seasons(load.load_usage(**kw)), ppa)
    stats = load.load_stats(**kw).rename(columns={"playerId": "id"})
    for df in (ppa, usage, stats):
        df["team"] = clean.norm_team(df.team)
        df["id"] = apply_aliases(df.id, aliases)
    return ppa, stats, usage


def team_level(conference: pd.Series) -> pd.Series:
    conf = conference.fillna("").astype(str)
    return pd.Series(np.where(conf.isin(FBS_CONFERENCES), "FBS",
                              np.where(conf.eq(""), "other", "FCS")),
                     index=conference.index, dtype="string")


def implied_plays(total: pd.Series, average: pd.Series) -> pd.Series:
    """total / average where |average| >= MIN_ABS_AVERAGE; 0 when there is no
    total; NaN when the rounded average can't pin the plays down, or when total
    and average disagree in sign (inconsistent row)."""
    plays = (total / average).where(average.abs() >= MIN_ABS_AVERAGE)
    plays = plays.where(~(plays < 0))
    none = total.fillna(0).eq(0) & (average.isna() | average.eq(0))
    return plays.mask(none, 0.0)


def box_plays(stats: pd.DataFrame) -> pd.DataFrame:
    """Per (id, season, team): pass ATT, rushing CAR, receptions REC, box_plays."""
    wanted = {("passing", "ATT"), ("rushing", "CAR"), ("receiving", "REC")}
    rows = stats[[pair in wanted for pair in zip(stats.category, stats.statType)]].copy()
    rows["stat"] = pd.to_numeric(rows.stat, errors="coerce")
    wide = (rows.pivot_table(index=["id", "season", "team"], columns="statType",
                             values="stat", aggfunc="sum")
                .reindex(columns=["ATT", "CAR", "REC"]).fillna(0))
    wide["box_plays"] = wide.sum(axis=1)
    return wide.reset_index()


def team_plays(box: pd.DataFrame) -> pd.Series:
    """Team offensive plays per (season, team): all pass attempts + carries."""
    return box.groupby(["season", "team"])[["ATT", "CAR"]].sum().sum(axis=1).rename("team_plays")


# --- Player seasons -------------------------------------------------------------

def season_position(listed: pd.Series, att: pd.Series, car: pd.Series,
                    rec: pd.Series) -> pd.Series:
    """The position a player actually played that season, from his play mix.

    The PPA table labels each row with a roster position that can be from
    another season (Donaven McCulley: Indiana QB in 2021, listed WR).
      - QB if he threw QB_MIN_ATTEMPTS+ passes, or more passes than carries and
        receptions (run-first option QBs still throw 25+).
      - A listed QB with zero pass attempts -> WR if receptions lead, RB with
        RB_MIN_CARRIES+ carries; otherwise (kneel-downs, a few carries) still QB.
      - Otherwise, and with no box stats, the listed position.
    """
    att, car, rec = (x.fillna(0) for x in (att, car, rec))
    passer = (att >= QB_MIN_ATTEMPTS) | ((att > car) & (att > rec))
    no_pass_qb = listed.eq("QB") & att.eq(0)
    out = listed.mask(passer, "QB")
    out = out.mask(no_pass_qb & (rec > 0) & (rec >= car), "WR")
    out = out.mask(no_pass_qb & (car > rec) & (car >= RB_MIN_CARRIES), "RB")
    return out.astype("string")


def build_player_seasons(ppa: pd.DataFrame, stats: pd.DataFrame):
    """One row per (player_id, season, team) PPA line with plays, position and
    level. Returns (player_seasons, box calibration factor by position)."""
    ps = pd.DataFrame({
        "player_id": ppa.id.astype("string"), "season": ppa.season.astype(int),
        "team": ppa.team, "listed_position": clean.position_group(ppa.position),
        "level": team_level(ppa.conference), "total_ppa": ppa["totalPPA.all"],
        "implied_plays": implied_plays(ppa["totalPPA.pass"], ppa["averagePPA.pass"])
        + implied_plays(ppa["totalPPA.rush"], ppa["averagePPA.rush"]),
    })
    if ps.duplicated(["player_id", "season", "team"]).any():
        raise ValueError("PPA has repeated (player_id, season, team) rows after aliasing")
    box = box_plays(stats).rename(columns={"id": "player_id"})
    ps = ps.merge(box, on=["player_id", "season", "team"], how="left")
    ps["position_group"] = season_position(ps.listed_position, ps.ATT, ps.CAR, ps.REC)

    both = ps.implied_plays.gt(0) & ps.box_plays.gt(0)
    factor = (ps[both].implied_plays / ps[both].box_plays).groupby(
        ps[both].position_group).median()
    calibrated = ps.box_plays * ps.position_group.map(factor).astype(float)
    ps["plays"] = ps.implied_plays.fillna(calibrated)
    ps["plays_source"] = np.where(ps.implied_plays.notna(), "implied",
                                  np.where(calibrated.notna(), "box", "none"))
    ps["plays"] = ps.plays.fillna(0.0)
    return ps.drop(columns=["ATT", "CAR", "REC"]), factor.round(3).to_dict()


def compare_plays_estimators(ps: pd.DataFrame, usage: pd.DataFrame,
                             box: pd.DataFrame) -> pd.DataFrame:
    """How well box-score plays and usage share x team plays match implied
    plays (exact for the PPA rows they cover), by position."""
    u = usage.rename(columns={"id": "player_id"})[["player_id", "season", "team",
                                                   "usage.overall"]]
    d = (ps.merge(u, on=["player_id", "season", "team"], how="left")
           .join(team_plays(box), on=["season", "team"]))
    d["usage_plays"] = d["usage.overall"] * d.team_plays
    d = d[d.implied_plays.gt(0) & d.box_plays.gt(0) & d.usage_plays.gt(0)]
    rows = []
    for pos, g in d.groupby("position_group"):
        for est in ["box_plays", "usage_plays"]:
            ratio = g[est] / g.implied_plays
            rows.append({"position": pos, "estimator": est, "n": len(g),
                         "median_ratio": ratio.median(),
                         "median_abs_error": (ratio - 1).abs().median(),
                         "log_corr": np.corrcoef(np.log(g[est]), np.log(g.implied_plays))[0, 1]})
    return pd.DataFrame(rows)


def _regulars(ps):
    return ps[ps.position_group.isin(MIN_PLAYS)
              & (ps.plays >= ps.position_group.map(MIN_PLAYS).astype(float))
              & ps.level.isin(["FBS", "FCS"])]


def replacement_levels(ps: pd.DataFrame) -> pd.Series:
    """Replacement PPA per play by (position_group, season, level)."""
    reg = _regulars(ps).assign(rate=lambda d: d.total_ppa / d.plays)
    return (reg.groupby(["position_group", "season", "level"]).rate
               .quantile(REPLACEMENT_QUANTILE).rename("repl"))


def add_par(ps: pd.DataFrame, repl: pd.Series) -> pd.DataFrame:
    """par = total_ppa - repl * plays (0 at level "other" or with no replacement)."""
    ps = ps.join(repl, on=["position_group", "season", "level"])
    ps["par"] = (ps.total_ppa - ps.repl * ps.plays).where(ps.repl.notna(), 0.0)
    return ps


def estimate_exchange_rate(ps: pd.DataFrame, n_boot=2000, seed=0) -> dict:
    """FCS -> FBS exchange rate k from players who were regulars at both levels
    in consecutive seasons (either direction).

    k = sum(w * fbs_rate) / sum(w * fcs_rate), where *_rate is PPA per play
    above that level's replacement and w is the harmonic mean of the two
    seasons' plays. Both directions select on a good (up) or bad (down)
    season, so regression to the mean biases k low: FCS gets discounted a bit
    too much, the conservative side. Falls back to FALLBACK_FCS_DISCOUNT when
    there are fewer than MIN_MOVERS pairs or the 95% CI reaches 0.
    """
    reg = _regulars(ps)
    reg = reg.assign(rate=(reg.par / reg.plays))[["player_id", "season", "level", "plays",
                                                   "rate"]]
    nxt = reg.assign(season=reg.season - 1)
    pairs = reg.merge(nxt, on=["player_id", "season"], suffixes=("_a", "_b"))
    pairs = pairs[pairs.level_a.ne(pairs.level_b)]
    up = pairs.level_a.eq("FCS")
    fcs = np.where(up, pairs.rate_a, pairs.rate_b)
    fbs = np.where(up, pairs.rate_b, pairs.rate_a)
    w = 2 / (1 / pairs.plays_a + 1 / pairs.plays_b)
    d = pd.DataFrame({"fcs": fcs, "fbs": fbs, "w": w.to_numpy(), "up": up.to_numpy()})

    out = {"n_pairs": len(d), "n_up": int(d.up.sum()), "n_down": int((~d.up).sum())}
    if len(d) == 0:
        return {**out, "k": FALLBACK_FCS_DISCOUNT, "ci": (np.nan, np.nan), "source": "fallback"}

    def k_of(x):
        return (x.w * x.fbs).sum() / (x.w * x.fcs).sum()

    rng = np.random.default_rng(seed)
    boots = [k_of(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    k, ci = k_of(d), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    usable = len(d) >= MIN_MOVERS and ci[0] > 0
    return {**out, "k_estimate": float(k), "ci": ci,
            "k": float(k) if usable else FALLBACK_FCS_DISCOUNT,
            "source": "movers" if usable else "fallback"}


def add_fbs_equivalent(ps: pd.DataFrame, k: float) -> pd.DataFrame:
    scale = ps.level.map({"FBS": 1.0, "FCS": k, "other": 0.0}).astype(float)
    return ps.assign(par_fbs=ps.par * scale)


def add_roles(ps: pd.DataFrame) -> pd.DataFrame:
    """role: 0 none, 1 rotation, 2 starter (see module docstring). Level
    "other" is always 0."""
    min_plays = ps.position_group.map(MIN_PLAYS).astype(float)
    starters = ps.position_group.map(STARTERS).astype(float)
    rank = ps.groupby(["season", "team", "position_group"]).plays.rank(ascending=False,
                                                                       method="min")
    regular = (ps.plays >= min_plays) & ps.level.isin(["FBS", "FCS"])
    ps = ps.assign(role=np.where(regular & (rank <= starters), 2, np.where(regular, 1, 0)))
    return ps


def build_production(aliases: pd.DataFrame, raw_dir=None):
    """Everything above, end to end. Returns (player_seasons, info) where info
    holds the exchange rate, replacement levels, box calibration and the plays
    estimator comparison for the report."""
    ppa, stats, usage = load_inputs(aliases, raw_dir)
    ps, calibration = build_player_seasons(ppa, stats)
    repl = replacement_levels(ps)
    ps = add_par(ps, repl)
    exchange = estimate_exchange_rate(ps)
    ps = add_roles(add_fbs_equivalent(ps, exchange["k"]))
    info = {"exchange": exchange, "replacement": repl, "box_calibration": calibration,
            "plays_comparison": compare_plays_estimators(ps, usage, box_plays(stats))}
    return ps, info
