# CFB Moneyball: Player Value Model

Goal: Find market inefficiencies in college football talent evaluation.
"Market price" = recruiting/portal star ratings and rankings.
"True value" = on-field production afterward (PPA/EPA, snaps, starts, honors, NFL draft).
Model predicts production from info known at signing; gaps vs market rank = value.

Data: CollegeFootballData.com API (key stored in .env as CFBD_API_KEY, never committed).
Scope: portal era 2021-2025, start with QB, RB, WR.
Stack: Python, pandas, scikit-learn/LightGBM, pytest.

I'm a student building this to learn and to show interviewers.
- Explain your plan before writing code.
- Keep changes small; one step at a time.
- Don't write the modeling code unless I ask; I want to write it myself.

## Git
- Never add a Co-Authored-By line (or any AI attribution) to commit messages.
