"""Master table: one row per acquisition (a recruit signing or a transfer).

    python -m cfb.master    # writes data/processed/master.parquet, master_report.txt

Rows (QB/RB/WR, seasons 2021-2025):
  recruit   a recruit from the 2021-2025 classes; acquiring team = committedTo
  transfer  a portal entry with a destination (not withdrawn, not gone pro);
            acquiring team = destination

Columns, by role:
  market price  market_stars, market_rating, market_unrated, market_tier
                (recruiting ratings for recruits, portal ratings for transfers;
                `ranking` is deliberately left out - see CLAUDE.md)
  features      known at signing only: position_group, qb_style, height, weight,
                home_state, recruit_type, hs_* (the high-school record, also for
                transfers when linked), origin/acquiring level, eligibility,
                years_since_hs, prior_* (transfers: production before the move).
                The college position is NOT a feature (used only to place ATHs).
  outcomes      A (main): par_window, FBS-equivalent PPA above replacement over
                WINDOW seasons, credited at any team; par_window_at_team.
                B: tier_window (0 none, 1 rotation, 2 starter), tier_window_fbs.
                C (side check): drafted_after, draft_round, draft_overall.
  bookkeeping   seasons_observed, window_complete, unmatched, match_method,
                nationwide_dependent, draft_observable.

No stat line in the window -> 0 (a bust). Unmatched transfers are kept at 0
with `unmatched` = True; drop them as a robustness check. nationwide_name
links are excluded unless include_nationwide=True (see cfb.processed).
"""

import argparse

import numpy as np
import pandas as pd

from cfb import clean, load
from cfb.processed import PROCESSED_DIR, load_id_aliases, load_players, load_portal_links
from cfb.production import build_production

SEASONS = (2021, 2025)
SCOPE = ("QB", "RB", "WR")
WINDOW = {"recruit": 3, "transfer": 2}
FIRST_DATA_SEASON = 2021


# --- Team levels ----------------------------------------------------------------

def level_lookup(ps: pd.DataFrame):
    """(team, season) -> level, and team -> its most common level."""
    def mode(s):
        return s.mode().iloc[0]
    by_season = ps.groupby(["team", "season"])["level"].agg(mode)
    by_team = ps.groupby("team")["level"].agg(mode)
    return by_season, by_team


def level_of(teams: pd.Series, seasons: pd.Series, lookup) -> pd.Series:
    """Level of each team in that season; else its usual level; else "other"
    (teams never in the PPA data: JUCO, D2, NAIA). NA team -> NA."""
    by_season, by_team = lookup
    keys = pd.MultiIndex.from_arrays([teams.astype(object), seasons.astype(int)])
    level = pd.Series(by_season.reindex(keys).to_numpy(), index=teams.index, dtype="string")
    level = level.fillna(teams.map(by_team).astype("string")).fillna("other")
    return level.mask(teams.isna()).astype("string")


# --- Acquisitions ---------------------------------------------------------------

def _recruit_extras(recruits_raw):
    r = recruits_raw.assign(recruit_id=recruits_raw.id.astype("string")).set_index("recruit_id")
    return r[["height", "weight", "stateProvince", "recruitType"]].rename(
        columns={"stateProvince": "home_state", "recruitType": "recruit_type"})


def recruit_acquisitions(players: pd.DataFrame, recruits_raw: pd.DataFrame) -> pd.DataFrame:
    r = players[players.in_recruits.astype(bool) & players.year.between(*SEASONS)
                & players.position_group.isin(SCOPE)]
    out = pd.DataFrame({
        "acquisition_id": "rec:" + r.recruit_id, "acquisition_type": "recruit",
        "person_id": r.person_id, "player_id": r.player_id, "recruit_id": r.recruit_id,
        "portal_key": pd.NA, "name": r.name, "acq_season": r.year.astype(int),
        "acquiring_team": r.committedTo, "origin_team": pd.NA,
        "position_group": r.position_group, "qb_style": r.qb_style,
        "market_stars": r.stars, "market_rating": r.rating, "market_tier": r.market_tier,
        "hs_class_year": r.year, "hs_stars": r.stars, "hs_rating": r.rating,
        "eligibility": pd.NA, "match_method": r.recruit_match_method,
        "nationwide_dependent": False,
    })
    return out.join(_recruit_extras(recruits_raw), on="recruit_id")


def transfer_acquisitions(links: pd.DataFrame, players: pd.DataFrame,
                          recruits_raw: pd.DataFrame) -> pd.DataFrame:
    t = links[links.season.between(*SEASONS) & links.destination.notna()
              & ~links.withdrawn.astype(bool) & ~links.went_pro.astype(bool)
              & links.position_group.isin(SCOPE)]
    hs = players[players.in_recruits.astype(bool)].set_index("recruit_id")
    out = pd.DataFrame({
        "acquisition_id": "portal:" + t.portal_key, "acquisition_type": "transfer",
        "person_id": t.person_id, "player_id": t.player_id, "recruit_id": t.recruit_id,
        "portal_key": t.portal_key, "name": t.firstName + " " + t.lastName,
        "acq_season": t.season.astype(int), "acquiring_team": t.destination,
        "origin_team": t.origin, "position_group": t.position_group,
        "qb_style": t.recruit_id.map(hs.qb_style),
        "market_stars": t.stars, "market_rating": t.rating, "market_tier": t.market_tier,
        "hs_class_year": t.recruit_id.map(hs.year), "hs_stars": t.recruit_id.map(hs.stars),
        "hs_rating": t.recruit_id.map(hs.rating), "eligibility": t.eligibility,
        "match_method": t.match_method, "nationwide_dependent": t.nationwide_dependent,
    })
    return out.join(_recruit_extras(recruits_raw), on="recruit_id")


# --- Outcomes -------------------------------------------------------------------

def _player_rows(acq, ps):
    rows = acq[["acquisition_id", "player_id", "acq_season", "acquiring_team",
                "acquisition_type"]].dropna(subset=["player_id"])
    return rows.merge(ps[["player_id", "season", "team", "level", "plays", "par_fbs", "role"]],
                      on="player_id")


def window_outcomes(acq: pd.DataFrame, ps: pd.DataFrame, last_season: int) -> pd.DataFrame:
    """Outcomes A and B over each acquisition's window, indexed by acquisition_id."""
    width = acq.acquisition_type.map(WINDOW).astype(int)
    m = _player_rows(acq, ps)
    m = m[(m.season >= m.acq_season)
          & (m.season < m.acq_season + m.acquisition_type.map(WINDOW).astype(int))]
    at_team = m.team.eq(m.acquiring_team)
    played = m.plays > 0

    def per_acq(frame, col, how):
        return frame.groupby("acquisition_id")[col].agg(how)

    out = pd.DataFrame(index=pd.Index(acq.acquisition_id, name="acquisition_id"))
    out["par_window"] = per_acq(m, "par_fbs", "sum")
    out["par_window_at_team"] = per_acq(m[at_team], "par_fbs", "sum")
    out["plays_window"] = per_acq(m, "plays", "sum")
    out["seasons_played"] = per_acq(m[played], "season", "nunique")
    out["tier_window"] = per_acq(m, "role", "max")
    out["tier_window_fbs"] = per_acq(m[m.level.eq("FBS")], "role", "max")
    out = out.fillna(0)
    for col in ["seasons_played", "tier_window", "tier_window_fbs"]:
        out[col] = out[col].astype(int)

    observed = (last_season - acq.acq_season + 1).clip(lower=0).clip(upper=width)
    out["seasons_observed"] = observed.to_numpy()
    out["window_complete"] = (observed == width).to_numpy()
    return out


def prior_production(acq: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    """Transfers: production before the move (known at transfer time).

    NaN when there is nothing to look at: recruits, unmatched transfers (their
    history is unknown, not zero), and 2021 transfers (2020 isn't in the data).
    """
    m = _player_rows(acq, ps)
    m = m[m.season < m.acq_season]
    g = m.groupby("acquisition_id")
    out = pd.DataFrame(index=pd.Index(acq.acquisition_id, name="acquisition_id"))
    out["prior_par"] = g.par_fbs.sum()
    out["prior_plays"] = g.plays.sum()
    out["prior_seasons_played"] = m[m.plays > 0].groupby("acquisition_id").season.nunique()
    out["prior_best_tier"] = g.role.max()
    out = out.fillna(0)
    known = (acq.acquisition_type.eq("transfer") & acq.player_id.notna()
             & acq.acq_season.gt(FIRST_DATA_SEASON)).to_numpy()
    out = out.where(pd.Series(known, index=out.index), np.nan)
    out["prior_seasons_available"] = np.where(
        acq.acquisition_type.eq("transfer"), acq.acq_season - FIRST_DATA_SEASON, np.nan)
    return out


def draft_outcomes(acq: pd.DataFrame, players_all: pd.DataFrame,
                   last_draft_year: int) -> pd.DataFrame:
    """Outcome C: drafted after the acquisition. Keyed by player_id, so it does
    not depend on how the recruit record was linked."""
    picks = (players_all[players_all.drafted.astype(bool) & players_all.player_id.notna()]
             .drop_duplicates("player_id").set_index("player_id"))
    year = acq.player_id.map(picks.draft_year)
    after = year.notna() & (year >= acq.acq_season + 1)
    width = acq.acquisition_type.map(WINDOW).astype(int)
    return pd.DataFrame({
        "drafted_after": after.to_numpy(),
        "draft_year": year.where(after).to_numpy(),
        "draft_round": acq.player_id.map(picks.draft_round).where(after).to_numpy(),
        "draft_overall": acq.player_id.map(picks.draft_overall).where(after).to_numpy(),
        "draft_observable": (acq.acq_season + width <= last_draft_year).to_numpy(),
    }, index=pd.Index(acq.acquisition_id, name="acquisition_id"))


# --- Assembly ---------------------------------------------------------------------

def build_master(players, players_all, links, recruits_raw, ps) -> pd.DataFrame:
    """players / links: analysis sample (cfb.processed loaders); players_all: the
    full players table (draft lookup by player_id); ps: player seasons."""
    acq = pd.concat([recruit_acquisitions(players, recruits_raw),
                     transfer_acquisitions(links, players, recruits_raw)], ignore_index=True)
    if acq.acquisition_id.duplicated().any():
        raise ValueError("duplicate acquisition_id")
    lookup = level_lookup(ps)
    acq["acquiring_level"] = level_of(acq.acquiring_team, acq.acq_season, lookup)
    acq["origin_level"] = level_of(acq.origin_team, acq.acq_season - 1, lookup)
    acq["market_unrated"] = acq.market_stars.isna()
    acq["years_since_hs"] = acq.acq_season - acq.hs_class_year
    acq["unmatched"] = acq.player_id.isna()

    last_season = int(ps.season.max())
    last_draft = pd.to_numeric(players_all.draft_year, errors="coerce").max()
    last_draft = last_season + 1 if pd.isna(last_draft) else int(last_draft)  # no picks loaded
    acq = (acq.join(window_outcomes(acq, ps, last_season), on="acquisition_id")
              .join(prior_production(acq, ps), on="acquisition_id")
              .join(draft_outcomes(acq, players_all, last_draft), on="acquisition_id"))
    return acq.reset_index(drop=True)


def build(include_nationwide=False, processed_dir=None, raw_dir=None):
    """Load everything and build (master, production info)."""
    kw = {} if raw_dir is None else {"raw_dir": raw_dir}
    aliases = load_id_aliases(processed_dir)
    ps, info = build_production(aliases, raw_dir)
    recruits_raw = clean.clean_recruits(load.load_recruits(**kw))
    master = build_master(
        load_players(processed_dir, include_nationwide),
        load_players(processed_dir, include_nationwide=True),
        load_portal_links(processed_dir, include_nationwide), recruits_raw, ps)
    return master, info


# --- Report -------------------------------------------------------------------------

def _rating_corr(df):
    d = df[df.window_complete & df.market_rating.notna()]
    return d.groupby("acquisition_type").apply(
        lambda g: pd.Series({"n": len(g), "spearman": g.market_rating.corr(
            g.par_window, method="spearman")}), include_groups=False)


def report(master, master_nw, info) -> str:
    ex = info["exchange"]
    lines = ["MASTER TABLE", f"rows: {len(master):,}  " + ", ".join(
        f"{k} {v:,}" for k, v in master.acquisition_type.value_counts().items())]
    lines.append(pd.crosstab([master.acquisition_type, master.position_group],
                             master.window_complete).rename(
        columns={True: "window complete", False: "incomplete"}).to_string())
    t = master[master.acquisition_type.eq("transfer")]
    lines += [f"unmatched (scored 0): recruits {int(master[master.acquisition_type.eq('recruit')].unmatched.sum()):,}, "
              f"transfers {int(t.unmatched.sum()):,} (of which destination in FBS/FCS: "
              f"{int((t.unmatched & t.acquiring_level.isin(['FBS', 'FCS'])).sum()):,})",
              f"nationwide_dependent transfer rows (recruit link removed): "
              f"{int(t.nationwide_dependent.sum()):,}", ""]

    lines += ["PLAYS ESTIMATE (vs implied plays = totalPPA / averagePPA)",
              info["plays_comparison"].round(3).to_string(index=False),
              f"box-score calibration factors (implied / box): {info['box_calibration']}", ""]
    repl = info["replacement"].groupby(level=["position_group", "level"]).mean().unstack()
    lines += ["REPLACEMENT PPA PER PLAY (mean over seasons)", repl.round(3).to_string(), ""]
    lines += ["FCS -> FBS EXCHANGE RATE",
              f"k = {ex['k']:.3f} (source: {ex['source']}; estimate "
              f"{ex.get('k_estimate', float('nan')):.3f}, 95% CI {ex['ci'][0]:.3f}-{ex['ci'][1]:.3f}, "
              f"{ex['n_pairs']} movers: {ex['n_up']} up, {ex['n_down']} down)", ""]

    complete = master[master.window_complete]
    lines += ["OUTCOME A (par_window) BY MARKET TIER, complete windows",
              complete.pivot_table(index="market_tier", columns="acquisition_type",
                                   values="par_window", aggfunc=["mean", "count"])
                      .round(1).to_string(), ""]
    lines += ["OUTCOME B (tier_window) SHARES, complete windows",
              pd.crosstab([complete.acquisition_type, complete.market_tier],
                          complete.tier_window, normalize="index").round(2).to_string(), ""]
    obs = master[master.draft_observable]
    lines += ["OUTCOME C (drafted_after), draft-observable rows",
              obs.groupby(["acquisition_type", "market_tier"]).drafted_after.agg(
                  ["mean", "size"]).round(3).to_string(), ""]

    main = _rating_corr(master)
    no_unmatched = _rating_corr(master[~(master.acquisition_type.eq("transfer")
                                         & master.unmatched)])
    nw = _rating_corr(master_nw)
    lines += ["ROBUSTNESS: Spearman(market_rating, par_window), complete windows",
              pd.concat({"main": main, "drop unmatched transfers": no_unmatched,
                         "include nationwide_name": nw}, axis=1).round(3).to_string()]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    master, info = build(include_nationwide=False)
    master_nw, _ = build(include_nationwide=True)
    master.to_parquet(PROCESSED_DIR / "master.parquet", index=False)
    master_nw.to_parquet(PROCESSED_DIR / "master_with_nationwide.parquet", index=False)
    text = report(master, master_nw, info)
    (PROCESSED_DIR / "master_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote master.parquet, master_with_nationwide.parquet, master_report.txt "
          f"to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
