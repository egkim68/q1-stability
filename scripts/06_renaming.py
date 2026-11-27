"""Step 6. Bound on renamed journals and the 100-journal audit sample.

A rename that comes with a new Scopus source identifier leaves a footprint in the SJR files: a discontinued
journal (absent from the 2025 file) from the same publisher, sharing at least one subject category, whose last
SJR year falls in or just before the sample journal's Scopus entry year. For each sample journal this script
records such "handovers" at lags of 0 to 6 years. Lags 3-6 give the background rate of unrelated
discontinuations; the excess at lags 0 and 1 over twice that background is an upper bound on likely renames.

It also draws the random sample of 100 RECENT journals (seed 20261124) that was checked by hand against
Scopus source pages; the results of that check are in results/renaming_audit_100_RECENT.csv.

Usage: python scripts/06_renaming.py build/state.pkl build/
"""
import os
import re
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

st = pickle.load(open(sys.argv[1], 'rb'))
OUT = sys.argv[2]
p = st['panel']
r = pd.read_pickle(os.path.join(OUT, 'journals.pkl'))

STOP = {'ltd', 'inc', 'gmbh', 'co', 'llc', 'limited', 'publishing', 'publishers', 'press', 'the', 'and', 'verlag',
        'b', 'v', 'bv', 'ag', 'sa'}


def publisher_key(s):
    s = re.sub(r'[^a-z ]', ' ', str(s).lower())
    return ' '.join(w for w in s.split() if w not in STOP)


def categories(s):
    return {re.sub(r'\s*\(Q\d\)\s*$', '', c).strip() for c in str(s).split(';') if c.strip()} if isinstance(s, str) else set()


g = p.groupby('Sourceid')
src = pd.DataFrame({'first': g.year.min(), 'last': g.year.max(), 'Publisher': g.Publisher.last(), 'Type': g.Type.last(),
                    'Title': g.Title.last(), 'cats': g.Categories.apply(lambda x: set().union(*[categories(v) for v in x]))})
src['pub'] = src.Publisher.map(publisher_key)
in25 = set(p.loc[p.year == 2025, 'Sourceid'])
ended = src[(src.Type == 'journal') & ~src.index.isin(in25)]
index = {}
for k, row in ended.iterrows():
    for c in row.cats:
        index.setdefault((row.pub, c), []).append((k, row['last']))

LAGS = range(7)
hits = {L: [] for L in LAGS}
pred = {}
for sid in r.index:
    me = src.loc[sid]
    f = int(r.at[sid, 'entry'])
    found = {L: set() for L in LAGS}
    if me.pub:
        for c in me.cats:
            for k, last in index.get((me.pub, c), []):
                L = f - last
                if L in found and k != sid:
                    found[L].add(k)
    for L in LAGS:
        hits[L].append(bool(found[L]))
    pred[sid] = sorted(found[0] | found[1])
r = r.join(pd.DataFrame(hits, index=r.index).add_prefix('lag'))

out = {}
groups = {'all journals': r, 'RECENT': r[r.cat == 'RECENT'], 'classifiable': r[r.cat != 'RECENT'],
          'classifiable rapid ascenders': r[(r.cat != 'RECENT') & r.rapid]}
for name, d in groups.items():
    rates = {L: d[f'lag{L}'].mean() * 100 for L in LAGS}
    background = np.mean([rates[L] for L in (3, 4, 5, 6)])
    out[name] = dict(n=len(d), **{f'lag{L}': round(v, 1) for L, v in rates.items()}, background=round(background, 1),
                     excess_lag0_1=round(rates[0] + rates[1] - 2 * background, 1))
    print(name, out[name])

r['ytq'] = r.firstq1 - r.entry
for name, m in {'Q1 within a year of entry': r.ytq <= 1, 'Q1 after 2+ years': r.ytq >= 2}.items():
    d = r[m]
    rates = {L: d[f'lag{L}'].mean() * 100 for L in LAGS}
    background = np.mean([rates[L] for L in (3, 4, 5, 6)])
    print(f'{name}: n = {len(d)}, excess = {rates[0] + rates[1] - 2 * background:.1f}')

cl = r[r.cat != 'RECENT']


def association(d):
    t = pd.crosstab(d.rapid, d.in_q1_now)
    chi = chi2_contingency(t, correction=False)[0]
    return dict(n=len(d), rapid_retained=round(d[d.rapid].in_q1_now.mean() * 100, 1),
                others_retained=round(d[~d.rapid].in_q1_now.mean() * 100, 1), chi2=round(chi, 1))


out['association_all'] = association(cl)
out['association_without_handovers'] = association(cl[~(cl.lag0 | cl.lag1)])
print('association, all classifiable:', out['association_all'])
print('association, excluding lag-0/1 handovers:', out['association_without_handovers'])

# break-even: renamed rapid ascenders that kept Q1 needed to erase the retention gap
n_r = int(cl.rapid.sum())
p_r, p_o = cl[cl.rapid].in_q1_now.mean(), cl[~cl.rapid].in_q1_now.mean()
x = (p_r - p_o) * n_r / (1 - p_o)
print(f'break-even: {x:.0f} renamed rapid ascenders ({x / n_r * 100:.1f}% of classifiable rapid ascenders)')
json.dump(out, open(os.path.join(OUT, 'renaming_bound.json'), 'w'), indent=1)

# audit sample
rng = np.random.default_rng(20261124)
recent = r[r.cat == 'RECENT']
sample = recent.loc[rng.choice(recent.index.values, 100, replace=False)].sort_values('Title')
rows = [dict(Sourceid=s, Title=row.Title, ISSN=row.Issn, Publisher=row.Publisher, Scopus_entry_year=int(row.entry),
             first_Q1_year=int(row.firstq1),
             handover_candidate='; '.join(f'{src.at[k, "Title"]} (Sourceid {k}, last SJR year {src.at[k, "last"]})' for k in pred[s][:3]),
             Scopus_source_page=f'https://www.scopus.com/sourceid/{s}')
        for s, row in sample.iterrows()]
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'audit_sample_RECENT_100.csv'), index=False)
print('audit sample written: 100 RECENT journals')
