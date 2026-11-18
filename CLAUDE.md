# CFB Moneyball: Player Value Model

Goal: Find market inefficiencies in college football talent evaluation.
"Market price" = recruiting/portal star ratings and rankings.
"True value" = on-field production afterward (PPA/EPA, snaps, starts, honors, NFL draft).
Model predicts production from info known at signing; gaps vs market rank = value.

Data: CollegeFootballData.com API (key stored in .env as CFBD_API_KEY, never committed).
Scope: portal era 2021-2025, start with QB, RB, WR.
Stack: Python, pandas, scikit-learn/LightGBM, pytest.

I'm a student building this to learn and to show interviewers.

## Working style
- Claude writes the code, including the modeling code.
- Explain the plan and the reasoning behind choices before writing code.
- Stop for my approval at major decisions (target definition, feature set, model
  choice, validation scheme, anything that changes scope or the data pulled).
  Small implementation details don't need approval; explain them afterwards.
- Keep changes small; one step at a time, one commit per step.

## Git
- Never add a Co-Authored-By line (or any AI attribution) to commit messages.

## Project status (as of 2026-11-18)

### Built
- `src/cfb/client.py`: CFBD API client. Bearer auth from .env, retries on 429/5xx
  (urllib3 2.x waits 0s/2s/4s), caches raw JSON in `data/raw/`, tracks
  `calls_remaining` from the `X-CallLimit-Remaining` header. Free tier: 1,000 calls/month.
- `src/cfb/fetch.py`: `python -m cfb.fetch [--run] [--years ...]`. Dry run by default,
  skips cached files, stops below a quota floor. The full pull is done (about 40 calls):
  recruits 2017-2025, portal / season PPA (all plays only) / usage (with and without
  garbage time) / season stats for 2021-2025, draft 2021-2026. No team-level stats.
- `notebooks/01_explore.ipynb`: data exploration and matching findings (section 6).
- `src/cfb/load.py`: cached JSON -> DataFrames, with ID columns as strings.
- `src/cfb/clean.py`: name/team normalization, position map, dedupes, PPA/usage split merge.
- `src/cfb/crosswalk.py`: `python -m cfb.crosswalk` writes `data/processed/`:
  `players.parquet` (one row per person), `portal_links.parquet` (one row per portal
  entry), `id_aliases.parquet` (merged split IDs), `coverage.txt`.
- `src/cfb/handcheck.py`: `python -m cfb.handcheck sample|summarize`.
  `annotations/handcheck_v1.csv` is labeled (100 rows; Claude's suggested verdicts are
  kept in separate columns, `verdict` holds the final labels).
- `src/cfb/processed.py`: loads crosswalk outputs with the sample rules (nationwide_name off
  by default).
- `src/cfb/production.py`: player-season plays, level, replacement level, PPA above
  replacement (PAR), FCS->FBS exchange rate, roles.
- `src/cfb/master.py`: `python -m cfb.master` writes `data/processed/master.parquet` (one row
  per acquisition), `master_with_nationwide.parquet` (robustness), `master_report.txt`.
  8,478 rows: 5,322 recruits (2021-25 classes), 3,156 transfers (2021-25), QB/RB/WR.
- Tests: 111 passing (`pytest`). Setup: `pip install -r requirements.txt` (includes `-e .`).

### Key decisions
- Market price = `rating` (continuous), not `ranking`. `ranking` is a national rank by
  rating with shared ranks for ties, and ~91 rows hold a wrong rank (`ranking_suspect`).
- Positions: PRO/DUAL -> QB (keep `qb_style` as a feature), APB/FB -> RB. ATH takes the
  college position (most seasons, ties -> most recent), which is used ONLY to assign a
  player to a group, never as a model feature (it isn't known at signing).
- Portal players with no stars/rating -> `market_tier = "unrated"`. Never imputed.
- PPA/usage: 253 exact duplicates (blank conference) dropped. 70 position splits merged:
  totalPPA summed, averagePPA recomputed from implied plays, usage left NaN (split usage
  values are not additive).
- One ID space: recruit athleteId = stats playerId = PPA id = draft collegeAthleteId.
  person_id = that ID, else `rec:` / `portal:` / `draft:` keys.
- `match_method`: athlete_id, name_team_season, name_variant, nationwide_name (recruits
  only), unmatched. Matching never guesses:
  ambiguous candidates stay unmatched; a player_id goes to at most one recruit
  (athlete_id wins; colliding name matches are all dropped).
- Negative athleteIds (646, 2017-2019 classes) are placeholders and treated as missing.
  Stale athleteIds with no stats are upgraded to a unique name match (`replaced_athlete_id`).
- Portal windows: destination stat season in [S, S+1], origin in [S-2, S-1], where portal
  `season` = the season played at the destination.
- Team aliases are a small hand-verified list (no fuzzy team matching). Accents are kept.
- Hand-check scope: unmatched QB/RB/WR recruits from the 2021-2023 classes, plus unmatched
  portal entrants with a destination. Rates are reweighted by stratum size.
- Hand-check result (2026-11-18, before fixes): failed-join rate portal 47% (CI 30-64%),
  recruits 21% (CI 7-34%). The labels were AI-assisted and agree with Claude's
  suggestions on 96/100 rows, so they are not an independent check of that heuristic.
- Honorary records: `EXCLUDED_RECRUITS` in clean.py (Tate Myre). No data signature
  exists for honorary ratings, so the list is hand-verified.
- Position is only a tiebreaker in name matches (recruit, portal, draft): a unique
  name + team + window candidate is accepted at any position. Strict for the
  recruit<->portal link and for nationwide_name.
- name_variant: only without an exact candidate, same team and window. One name part exact,
  the other close (shared surname word + prefix/nickname/0.8 first name, or same first name
  + 0.85 surname). Exact beats variant for the same ID.
- nationwide_name: name unique among all stat lines, first season >= class year,
  compatible position, unclaimed. Lower confidence: EXCLUDED from the main analysis by
  default via `cfb.processed.load_players()` / `load_portal_links()`
  (`include_nationwide=True` for the robustness check). Excluded recruits are dropped,
  not counted as busts; their portal links lose the recruit link (`nationwide_dependent`).
- Split IDs merged: same name + team, no shared season, at most 1 empty season between,
  compatible positions, and not two different recruits' athleteIds (26 merged).
  Production tables must be remapped with `id_aliases.parquet`.

### Target and master table (decided 2026-11-18)
- Grain: one row per acquisition. Recruit (2021-25 class, team = committedTo) or transfer
  (portal entry 2021-25 with a destination, not withdrawn / gone pro, team = destination).
- Outcome A (main): `par_window` = FBS-equivalent PPA above replacement, summed over 3
  seasons (recruits) or 2 (transfers), credited at ANY team; `par_window_at_team` too.
  No stat line -> 0. Use `window_complete` rows for training (recruits 2021-23 classes,
  transfers 2021-24).
- Outcome B (secondary): `tier_window` 0 none / 1 rotation / 2 starter. Outcome C (side
  check): `drafted_after`, valid where `draft_observable`.
- Plays: implied plays (totalPPA / averagePPA, |avg| >= 0.05; 90% of rows), else box score
  ATT + CAR + REC x per-position calibration. Usage share x team plays was worse at every
  position, WR included (log-corr 0.78 vs 0.94), so it is not used.
- Season position from the play mix (25+ pass attempts = QB), since PPA position labels
  can come from another season.
- Replacement = 20th percentile PPA/play among regulars (QB 100, RB 50, WR/TE 20 plays),
  per position, season, level. D2/NAIA seasons = 0.
- FCS -> FBS exchange rate estimated from 176 level movers: k = 0.48 (95% CI 0.35-0.62),
  likely biased low (both directions select on an extreme season). Fallback 0.5 if < 50
  movers or CI reaches 0.
- Unmatched transfers are kept at 0 with `unmatched`; robustness check drops them.
  Transfer `prior_*` features are NaN for unmatched entries and for 2021 (no 2020 data).
- First look (complete windows): Spearman(market_rating, par_window) = 0.23 for recruits
  and 0.24 for transfers; stable when dropping unmatched transfers (0.22) or adding
  nationwide_name (0.23).

### Coverage (from coverage.txt; before hand-check fixes in parentheses)
- Recruits from the 2021-2025 classes with any stat line: QB 51.7% (48.2), RB 65.8% (62.4),
  WR 60.5% (58.0).
- Portal entrants with a destination, matched: QB 83.6% (77.7), RB 92.0% (86.2),
  WR 86.4% (83.7).
- Hand-check: 26 of 42 labeled failed joins now matched, 0 of 54 busts matched.
- Draft: 1,549 of 1,551 picks matched by ID.

### Open issues
- Portal residual failed-join rate among still-unmatched entrants is about 30% (from the
  labels): wrong portal destinations (Damante, Costelli, D.J. Matthews), personal nicknames
  (Chip, Woody, KC, Trey), and split IDs across teams or in the same season (Hooks,
  Adkins, Aaron Young), which the same-team merge rule does not cover. Recruits: about 2%.
- 154 `link_conflict`s remain.
- nationwide_name precision is only spot-checked (common names like Trey Adams are
  uncertain). Consider a second hand-check sample of matched rows.
- Unrated recruits average more PAR than 3-stars (7.1 vs 6.2, complete windows). Check
  who they are (JUCO / late signees?) before modeling.
- 300 unmatched transfers to FBS/FCS teams are scored 0 though ~30% may have played.
- Not yet decided: feature set, model choice, validation scheme, how to price transfers
  with no rating (`market_unrated` is a flag for now).
