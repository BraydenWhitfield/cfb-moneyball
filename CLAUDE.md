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
  recruits 2017-2025, portal / season PPA / usage (with and without garbage time) /
  season stats for 2021-2025, draft 2021-2026.
- `notebooks/01_explore.ipynb`: data exploration and matching findings (section 6).
- `src/cfb/load.py`: cached JSON -> DataFrames, with ID columns as strings.
- `src/cfb/clean.py`: name/team normalization, position map, dedupes, PPA/usage split merge.
- `src/cfb/crosswalk.py`: `python -m cfb.crosswalk` writes `data/processed/`:
  `players.parquet` (one row per person), `portal_links.parquet` (one row per portal
  entry), `coverage.txt`.
- `src/cfb/handcheck.py`: `python -m cfb.handcheck sample|summarize`. A stratified
  sample of unmatched players is in `annotations/handcheck_v1.csv`, waiting for
  my labels (bust / failed_join / unsure).
- Tests: 63 passing (`pytest`). Setup: `pip install -r requirements.txt` (includes `-e .`).

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
- `match_method`: athlete_id, name_team_season, unmatched. Matching never guesses:
  ambiguous candidates stay unmatched; a player_id goes to at most one recruit
  (athlete_id wins; colliding name matches are all dropped).
- Negative athleteIds (646, 2017-2019 classes) are placeholders and treated as missing.
  Stale athleteIds with no stats are upgraded to a unique name match (`replaced_athlete_id`).
- Portal windows: destination stat season in [S, S+1], origin in [S-2, S-1], where portal
  `season` = the season played at the destination.
- Team aliases are a small hand-verified list (no fuzzy team matching). Accents are kept.
- Hand-check scope: unmatched QB/RB/WR recruits from the 2021-2023 classes, plus unmatched
  portal entrants with a destination. Rates are reweighted by stratum size.

### Coverage (from coverage.txt)
- Recruits from the 2021-2025 classes with any stat line: QB 48.2%, RB 62.4%, WR 58.0%.
- Portal entrants with a destination, matched: QB 77.7%, RB 86.2%, WR 83.7%.
- Draft: 1,549 of 1,551 picks matched by ID.

### Open issues
- Duplicate IDs inside the stats data: one player can have two playerIds (e.g. Kevric
  Wiggins at Jackson State), which can split one person into two rows. These drive many of
  the 150 remaining `link_conflict`s.
- QB/punter position rule: `positions_compatible` rejects QB vs ST, which blocked e.g.
  Jack Dawson (portal QB who appears as a punter). Revisit after the hand-check.
- Nicknames (Cam/Cameron) aren't handled. Fuzzy name matching depends on the hand-check's
  failed-join rate.
- Hand-check labels not done yet. Next: label, run `summarize`, decide on fuzzy matching.
- Not yet decided: target definition for the model, how to price transfers with no rating,
  and how to handle recent classes with few seasons observed (censoring).
