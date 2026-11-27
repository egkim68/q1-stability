"""Step 1. Build the panel, the post-2004 entry cohort, the broader sample, the QTT categories and the transition pairs.

Usage: python scripts/01_build_sample.py data/ build/state.pkl

Rules
  Cohort: sources of type "journal" in the 2025 file whose reconstructed entry year (minimum year in the
          Coverage field of the 2025 file) is 2004 or later.
  Sample: cohort journals whose first Q1 year (2004-2025) is no earlier than the entry year.
  RECENT: first Q1 year 2021 or later. in_Q1_now: Q1 in the 2025 file.
  QTT:    SUSTAINED (Q1 now, 0 dips), RECOVERING (Q1 now, 1 dip), FLUCTUATING (Q1 now, 2+ dips), DECLINING (not Q1 now).
  Areas:  taken from each journal's earliest annual file.
  Pairs:  classifiable (non-RECENT) journals, post-attainment years, literally adjacent calendar years only.
"""
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
COLS = ['Sourceid', 'Title', 'Type', 'Issn', 'Publisher', 'SJR Best Quartile', 'Coverage', 'Categories', 'Areas']

frames = []
for y in range(2004, 2026):
    df = pd.read_csv(os.path.join(D, f'sjr{y}.csv'), sep=';', dtype=str, usecols=lambda c: c in COLS)
    df = df.loc[:, ~df.columns.duplicated()]
    df['year'] = y
    frames.append(df)
p = pd.concat(frames, ignore_index=True)
p['Sourceid'] = p['Sourceid'].astype(np.int64)
p = p.sort_values(['Sourceid', 'year'])


def parse_coverage(s):
    """Coverage string such as '2004-2010, 2012, 2014-2025' -> set of years."""
    years = set()
    if not isinstance(s, str):
        return years
    for part in s.split(','):
        part = part.strip()
        m = re.fullmatch(r'(\d{4})\s*-\s*(\d{4})', part)
        if m:
            years.update(range(int(m[1]), int(m[2]) + 1))
        elif re.fullmatch(r'\d{4}', part):
            years.add(int(part))
    return years


in25 = set(p.loc[p.year == 2025, 'Sourceid'])
last = p.groupby('Sourceid').last()
first = p.groupby('Sourceid').first()
j = last[['Type', 'Coverage', 'Title', 'Issn', 'Publisher']].copy()
j['entry'] = j['Coverage'].map(lambda s: min(parse_coverage(s)) if parse_coverage(s) else np.nan)
cohort = j[(j.Type == 'journal') & (j.entry >= 2004) & j.index.isin(in25)].copy()

p['q'] = p['SJR Best Quartile'].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4})
obs = p.dropna(subset=['q'])
obs = obs[obs.Sourceid.isin(cohort.index)]
seqs = {sid: dict(zip(g.year, g.q.astype(int))) for sid, g in obs.groupby('Sourceid')}

rows = []
excluded = 0
for sid, e in cohort.entry.items():
    s = seqs.get(sid, {})
    q1_years = [y for y, v in s.items() if v == 1]
    if not q1_years:
        continue
    if min(q1_years) < e:
        excluded += 1  # first Q1 listing predates the reconstructed entry year
        continue
    f = min(q1_years)
    post = [(y, s[y]) for y in sorted(s) if y >= f]
    qs = [v for _, v in post]
    ndips = sum(v != 1 for v in qs)
    now = s.get(2025) == 1
    best = run = 0
    prev = None
    for y, v in post:
        run = run + 1 if (v == 1 and prev is not None and y == prev + 1 and run > 0) else (1 if v == 1 else 0)
        best = max(best, run)
        prev = y
    if f >= 2021:
        cat = 'RECENT'
    elif not now:
        cat = 'DECLINING'
    elif ndips == 0:
        cat = 'SUSTAINED'
    elif ndips == 1:
        cat = 'RECOVERING'
    else:
        cat = 'FLUCTUATING'
    rows.append(dict(sid=sid, firstq1=f, entry=e, cat=cat, ndips=ndips, longest_q1_run=best,
                     sustained3=best >= 3, nobs=len(qs), in_q1_now=now))
r = pd.DataFrame(rows).set_index('sid')
r = r.join(cohort[['Title', 'Issn', 'Publisher']])
r['areas_first'] = first.loc[r.index, 'Areas']

pairs = []
for sid, row in r[r.cat != 'RECENT'].iterrows():
    post = [(y, v) for y, v in sorted(seqs[sid].items()) if y >= row.firstq1]
    for (y0, v0), (y1, v1) in zip(post, post[1:]):
        if y1 == y0 + 1:
            # dur: position of the origin year in the post-attainment window (1 = year of first attainment)
            pairs.append((sid, y0, y1, v0, v1, y0 - row.firstq1 + 1))
P = pd.DataFrame(pairs, columns=['sid', 'y0', 'y1', 'fr', 'to', 'dur'])

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
pickle.dump(dict(panel=p, journals=r, seqs=seqs, pairs=P, cohort=cohort), open(OUT, 'wb'))
print('cohort', len(cohort), '| sample', len(r), '| excluded (first Q1 before entry)', excluded,
      '| QTT', r.cat.value_counts().to_dict())
print('never sustained Q1 for 3 consecutive years:', int((~r.sustained3).sum()))
