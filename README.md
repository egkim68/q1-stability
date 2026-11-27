# q1-stability

Code for Kim, E. *How Stable Is Q1 Status? Field-Dependent Persistence in SJR Journal Quartiles*.

## Data

The annual SJR files (2004–2025) are available from [SCImago Journal & Country Rank](https://www.scimagojr.com). Download them and save them as `data/sjr2004.csv` … `data/sjr2025.csv`.

## Run

```
pip install -r requirements.txt
python scripts/01_build_sample.py data/ build/state.pkl
python scripts/02_metrics_markov.py build/state.pkl build/
python scripts/03_fields.py build/
python scripts/04_rapid_ascent.py build/state.pkl build/
python scripts/05_markov_checks.py build/state.pkl build/markov_checks.json
python scripts/06_renaming.py build/state.pkl build/
```

`results/renaming_audit_100_RECENT.csv` is the manual check of 100 journals against Scopus source pages.
