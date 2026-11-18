"""Hand-check unmatched players: are they busts or failed joins?

    python -m cfb.handcheck sample      # write annotations/handcheck_v1.csv
    python -m cfb.handcheck summarize   # estimate the failed-join rate

`sample` draws a stratified random sample (fixed seed) of unmatched recruits
and portal entrants, with the 3 closest stat-line names at their schools. You
fill in `verdict` (bust / failed_join / unsure) and `notes` by checking each
player on 247/ESPN. `summarize` turns the verdicts into a failed-join rate with
a 95% confidence interval, reweighted by how big each stratum really is.

Labels are hand work that code can't regenerate, so the CSV lives in the
committed annotations/ folder, and `sample` refuses to overwrite it.
"""

import argparse
import difflib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.client import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "annotations" / "handcheck_v1.csv"
SCOPE = ["QB", "RB", "WR"]
# 2021-2023 classes: at least two seasons inside our 2021-2025 stats window, so
# "no stat line" isn't just "hasn't played yet" or "played before 2021".
RECRUIT_CLASSES = (2021, 2023)
VERDICTS = ("bust", "failed_join", "unsure")
SEED = 2026


def tier_band(tier: pd.Series) -> pd.Series:
    """Collapse star tiers into 3 bands with enough players each."""
    return tier.map({"5": "4-5", "4": "4-5", "3": "3"}).fillna("2/unrated")


def unmatched_population(players: pd.DataFrame, portal_links: pd.DataFrame) -> pd.DataFrame:
    """Everyone eligible for the hand-check, one row each, with a `stratum`."""
    rec = players[players.in_recruits.astype(bool)
                  & players.recruit_match_method.eq("unmatched")
                  & players.position_group.isin(SCOPE)
                  & players.year.between(*RECRUIT_CLASSES)]
    rec = pd.DataFrame({
        "source": "recruit", "key": rec.recruit_id, "person_id": rec.person_id,
        "name": rec.name, "position_group": rec.position_group,
        "year_or_season": rec.year, "teams": rec.committedTo,
        "tier": rec.market_tier.fillna("unrated"),
        "flags": np.where(rec.ambiguous, "ambiguous", np.where(rec.conflict, "conflict", "")),
    })
    pl = portal_links[portal_links.match_method.eq("unmatched")
                      & portal_links.destination.notna()
                      & ~portal_links.withdrawn.astype(bool)
                      & ~portal_links.went_pro.astype(bool)
                      & portal_links.position_group.isin(SCOPE)]
    pl = pd.DataFrame({
        "source": "portal", "key": pl.portal_key, "person_id": pl.person_id,
        "name": pl.firstName + " " + pl.lastName, "position_group": pl.position_group,
        "year_or_season": pl.season, "teams": pl.origin + " -> " + pl.destination,
        "tier": pl.market_tier.fillna("unrated"),
        "flags": np.where(pl.ambiguous, "ambiguous", ""),
    })
    pop = pd.concat([rec, pl], ignore_index=True)
    pop["stratum"] = pop.source + "|" + pop.position_group + "|" + tier_band(pop.tier)
    return pop


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Spread n draws as evenly as possible over strata, never more than a
    stratum holds; leftovers go to strata with room. E.g. sizes [100, 3, 50],
    n=12 -> [5, 3, 4]."""
    take = pd.Series(0, index=sizes.index)
    remaining = n
    while remaining > 0:
        open_ = take < sizes
        if not open_.any():
            break
        share = max(remaining // int(open_.sum()), 1)
        for s in sizes.index[open_]:
            add = min(share, sizes[s] - take[s], remaining)
            take[s] += add
            remaining -= add
            if remaining == 0:
                break
    return take


def closest_candidates(name: str, teams: list[str], stat_players: pd.DataFrame, k=3) -> list[str]:
    """The k most similar names among stat lines at these teams (any season)."""
    from cfb.clean import norm_name
    key = norm_name(pd.Series([name])).iloc[0]
    pool = stat_players[stat_players.team.isin(teams)]
    if pool.empty:
        return ["(no stat lines at " + " / ".join(teams) + " in our data)"]
    people = (pool.sort_values("stat_season")
                  .groupby("player_id")
                  .agg(name_key=("name_key", "last"), player_name=("player_name", "last"),
                       team=("team", "last"), seasons=("stat_season", lambda s: f"{s.min()}-{s.max()}"),
                       pos=("stat_position", "last")))
    people["score"] = [difflib.SequenceMatcher(None, key, other).ratio()
                       for other in people.name_key]
    top = people.sort_values("score", ascending=False).head(k)
    return [f"{r.player_name} | {r.team} {r.seasons} {r.pos} | id {pid} | {r.score:.2f}"
            for pid, r in top.iterrows()]


def draw_sample(population: pd.DataFrame, stat_players: pd.DataFrame, n=100,
                seed=SEED) -> pd.DataFrame:
    """Stratified sample: n/2 per source, spread evenly over that source's strata."""
    sizes = population.stratum.value_counts().sort_index()
    takes = []
    for source in ["recruit", "portal"]:
        src = sizes[sizes.index.str.startswith(source + "|")]
        takes.append(allocate(src, n // 2 if source == "recruit" else n - n // 2))
    take = pd.concat(takes)

    parts = [population[population.stratum == s].sample(int(k), random_state=seed)
             for s, k in take.items() if k > 0]
    sample = pd.concat(parts).sort_values(["source", "stratum", "name"]).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    sample["stratum_population"] = sample.stratum.map(sizes)
    sample["stratum_sampled"] = sample.stratum.map(take)

    for i in range(3):
        sample[f"candidate_{i + 1}"] = ""
    for idx, row in sample.iterrows():
        teams = [t.strip() for t in str(row.teams).split("->")]
        for i, text in enumerate(closest_candidates(row["name"], teams, stat_players)):
            sample.loc[idx, f"candidate_{i + 1}"] = text
    sample["verdict"] = ""
    sample["notes"] = ""
    return sample


def wilson(successes: int, n: int, z=1.96):
    """Wilson score 95% interval for a proportion (behaves well near 0 and 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def stratified_rate(labeled: pd.DataFrame, z=1.96):
    """Failed-join rate weighted by stratum population, with a normal-approx CI.

    Each stratum's rate is weighted by its share of the population (among
    strata that have labels). For the variance, a stratum's rate is nudged
    toward 1/2 ((x + 0.5) / (n + 1)) so a stratum of all-busts doesn't claim
    zero uncertainty.
    """
    g = labeled.groupby("stratum").agg(n=("failed", "size"), x=("failed", "sum"),
                                       N=("stratum_population", "first"))
    w = g.N / g.N.sum()
    p = (w * g.x / g.n).sum()
    p_tilde = (g.x + 0.5) / (g.n + 1)
    se = math.sqrt((w ** 2 * p_tilde * (1 - p_tilde) / g.n).sum())
    return p, max(0.0, p - z * se), min(1.0, p + z * se)


def summarize(path=DEFAULT_PATH) -> str:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"verdict": "string"})
    df["verdict"] = df.verdict.fillna("").str.strip().str.lower()
    bad = df[~df.verdict.isin(VERDICTS + ("",))]
    if len(bad):
        raise ValueError(f"Unknown verdicts in rows {bad.sample_id.tolist()}: "
                         f"{sorted(set(bad.verdict))}. Use one of {VERDICTS}.")

    # Records later found not to be real recruits leave both the sample and
    # their stratum's population count.
    from cfb.clean import EXCLUDED_RECRUITS
    keys = df["key"].astype(str) if "key" in df else pd.Series("", index=df.index)
    gone = df.source.eq("recruit") & keys.isin(EXCLUDED_RECRUITS)
    df["stratum_population"] -= df.stratum.map(df[gone].stratum.value_counts()).fillna(0).astype(int)
    df = df[~gone]

    lines = [f"{path}: {len(df)} rows, {df.verdict.ne('').sum()} labeled "
             f"({df.verdict.eq('unsure').sum()} unsure, excluded from rates)"
             + (f"; {int(gone.sum())} excluded record(s) dropped" if gone.any() else "")]
    labeled = df[df.verdict.isin(["bust", "failed_join"])].assign(
        failed=lambda d: d.verdict.eq("failed_join"))
    if labeled.empty:
        return "\n".join(lines + ["No bust/failed_join labels yet."])

    for source, part in labeled.groupby("source"):
        p, lo, hi = stratified_rate(part)
        lines.append(f"\n{source}: failed-join rate {p:.0%} (95% CI {lo:.0%}-{hi:.0%}), "
                     f"weighted by stratum size; {len(part)} labeled")
        for pos, sub in part.groupby("position_group"):
            lo, hi = wilson(int(sub.failed.sum()), len(sub))
            lines.append(f"  {pos}: {sub.failed.sum()}/{len(sub)} failed joins "
                         f"(unweighted, 95% CI {lo:.0%}-{hi:.0%})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="draw a new sample CSV")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", type=Path, default=DEFAULT_PATH)
    s.add_argument("--force", action="store_true", help="overwrite an existing file")
    m = sub.add_parser("summarize", help="summarize filled-in verdicts")
    m.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    if args.command == "summarize":
        print(summarize(args.path))
        return
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists (it may hold your labels). "
                         "Use --force or a new --out path.")
    from cfb.crosswalk import build_all
    players, portal_links, stat_players = build_all()
    population = unmatched_population(players, portal_links)
    sample = draw_sample(population, stat_players, n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False, encoding="utf-8-sig")  # -sig: Excel shows accents
    print(f"Population: {len(population):,} unmatched "
          + ", ".join(f"{k} {v:,}" for k, v in population.source.value_counts().items()))
    print(f"Wrote {len(sample)} rows to {args.out}")


if __name__ == "__main__":
    main()
