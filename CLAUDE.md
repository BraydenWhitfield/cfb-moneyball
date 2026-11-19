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
- `src/cfb/modeling/`: `data.py` (populations, features, splits), `evaluate.py` (metrics,
  paired bootstrap), `linear.py` (ridge), `gbm.py` (LightGBM + TreeSHAP), `explain.py`
  (SHAP summaries), `run.py` (`python -m cfb.modeling.run`: all learners x feature sets ->
  `data/processed/models/report.txt`), `prior_plays.py` (`python -m
  cfb.modeling.prior_plays` -> `prior_plays_report.txt`). Full run takes ~10 minutes.
- `src/cfb/modeling/groups.py` + `group_analysis.py` (`python -m cfb.modeling.group_analysis`
  -> `groups_report.txt`, `group_residuals.parquet`): recruit residuals vs baseline_plus,
  EB shrinkage, persistence, power curves, permutation + BH FDR, coach movers.
- Raw data also has `/coaches` 2020-2025 (FBS only) and `/teams` (2025 locations): 7 more
  API calls, 953 left that month (2026-11).
- Master table now includes TEs (for the group analysis only; models stay QB/RB/WR via
  `MODEL_POSITIONS`). TE PPA misses blocking value.
- Tests: 139 passing (`pytest`). Setup: `pip install -r requirements.txt` (includes `-e .`).

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
- 300 unmatched transfers to FBS/FCS teams are scored 0 though ~30% may have played.
- Unrated-vs-3-star FCS edge (+2.6 PAR, CI 0.1-5.2) did not survive out of time:
  `unrated_fcs` gets ~0.2% of SHAP and the full recruit model doesn't beat baseline_plus.
- Next: the write-up (no more new analyses, decided 2026-11-18). Later, when 2026 data
  arrives: confirm the QB prior-plays signal (a third test year; 2025 gets a complete
  2-season window).

### Modeling (decided 2026-11-18; code in `src/cfb/modeling/`)
- Unrated recruits looked better than 3-stars (7.1 vs 6.2) because of SELECTION: unrated
  records are added in bulk after signing (IDs in one late block, no ranking), mostly
  FCS / academy / Ivy signing lists, so every one has a school. 13.5% of 3-stars never
  signed with a covered school and score 0. With a school: 3-star 7.19 vs unrated 7.10.
- So the recruit model population DROPS recruits with no committedTo (nobody acquired
  them): 559 complete-window rows.
- Separate models for recruits (train 2021-22 classes, test 2023) and transfers (train
  2021-23, test 2024). Complete windows only. Squared-error loss (target can be negative,
  so no Tweedie).
- Feature sets: baseline = rating + unrated + position (transfers add stars and
  rating_missing: ~40% of rated transfers have stars but no rating); baseline_plus = +
  acquiring level (the main test); full = + all signing-time features. No class year.
- Same learner across feature sets (ridge, then LightGBM), tuned by grouped CV on
  person_id inside the training years. Test year scored once.
- Metrics: Spearman (pooled and within position) and within-tier top-20% lift (tiers 2,
  3, 4-5, unrated), bootstrap CIs and paired full-minus-baseline differences.
  Transfers reported on all test rows, new players only, and matched only.
- Added during the build: `tier_pos_lift` (lift inside tier x position cells; pooled tier
  lift rewards putting QBs first). Two transfer leaks closed: `prior_seasons_available`
  (= season - 2021, a disguised class year) is not a feature, and unmatched post-2021
  transfers get prior = 0 ("no linked history") instead of a missing flag (unmatched rows
  score 0 by construction). Some leak remains: the same failed join hides history and
  outcome.
- Test rows for transfer claims: prefer ORIGIN-matched entries (identity from pre-transfer
  data). "Matched only" partly selects on the outcome (destination matches need a stat
  line there); "all" includes unmatched rows (prior 0, outcome 0 by construction).

### Modeling results (2026-11-18; numbers in data/processed/models/*.txt)
Conclusion so far: NO robust market inefficiency. Signing-time features beyond price +
position + level don't improve out-of-time ranking within position. One candidate signal:
QB transfers' prior playing time.
- Recruits (test 2023 class, n=924): nothing beats baseline_plus. Full minus baseline_plus
  Spearman: ridge -0.013 [-0.040, +0.016], LightGBM -0.000 [-0.044, +0.043]. Best model
  overall is ridge:baseline_plus (Spearman 0.154 [0.085, 0.222]). LightGBM < ridge.
  SHAP puts 16% on home_area (TX +4.7, GA -3.5), which doesn't help out of time: noise.
- Transfers (test 2024, n=790): full beats baseline_plus on POOLED Spearman (ridge +0.056
  [+0.013, +0.097], LightGBM +0.065 [+0.014, +0.113]; holds for new players only), but
  not within position (~+0.02, CI spans 0), not on tier_pos_lift (-2 to -4, n.s.), and
  not on matched-only rows (ridge +0.033 [-0.012, +0.077]). Likely cross-position
  ordering and the residual unmatched leak, not underpricing.
- Transfer SHAP (LightGBM full): prior_plays 28%, position 25%, rating 14%, prior_par 11%,
  hs_rating 5%; dual-threat QB +5.5; moving down a level -0.8.
- Direct prior-plays test (`prior_plays.py`; per position, OLS controls = portal rating,
  stars, rating_missing, unrated, destination level; + log(1 + prior plays); trained
  2022-23 since 2021 has no prior data). Origin-matched rows, Spearman gain:
  - PRIMARY 2-season target, test 2024: QB +0.075 [-0.008, +0.165] p=0.04; RB -0.090
    [-0.222, +0.031]; WR +0.058 [-0.040, +0.163].
  - SECONDARY 1-season target (different window), 2024: QB +0.119 [-0.023, +0.254];
    2025 (train 2022-24): QB +0.117 [-0.005, +0.241] p=0.03, RB +0.064, WR +0.008.
  - In-sample the prior-plays coefficient is positive with CI > 0 at every position
    (QB ~ +10 to +15 PAR per SD), but out of time only QBs are consistent: positive in
    both test years, each CI just touching 0. RB flips sign. Several comparisons were
    run, so treat the QB result as suggestive, not established.

### Group analysis results (2026-11-18; `groups_report.txt`)
Question: do some groups of recruits consistently beat their rating? Residual = value -
ridge baseline_plus prediction (2021-22 cross-fitted, 2023 out of time), winsorized at the
1st/99th percentiles; QB/RB/WR/TE recruits with a school (1,997 + 1,116). EB shrinkage
toward 0 (DerSimonian-Laird tau2). Coach = most games in the class's first season (FBS
only, 77% of recruits); 448 transition classes.
- NOTHING PERSISTED. Correlation of 2021-22 shrunk effects with 2023 residuals: school
  +0.016 [-0.039, +0.072] (tau ~ 0: no between-school variance even in-sample); school x
  position -0.007 [-0.083, +0.066]; coach -0.004 [-0.072, +0.061] (-0.009 without
  transition classes). 2021 -> 2022 and raw-residual versions are also null.
- Power: with the real group sizes, true group effects with SD 5 PAR would have been
  detected 97% (school), 77% (school x position), 92% (coach) of the time; SD 3 only
  18-46%. So effects of ~5+ PAR are ruled out; small ones (SD <= 3) can't be.
- In-sample tau for school x position (4.5) and coach (4.1) did not persist: it reflects
  noise and heavy tails, not real group differences.
- In-state minus out-of-state: -1.45 [-3.32, +0.41] (2021-22), -0.45 [-2.69, +1.74]
  (2023). No effect.
- FDR (BH, q < 0.10, permutation p within class x position x level): one call-out,
  Boise State / Andy Avalos (same 12 recruits): +32 PAR in 2021-22 (q = 0.07 school, 0.095
  coach), then -5.6 in 2023 (n=8). Did not persist. No school x position cell survives.
- Coach movers (EXPLORATORY): 12 FBS coaches with classes at two schools; correlation of
  effects -0.04 [-0.84, +0.79]. Uninformative.
