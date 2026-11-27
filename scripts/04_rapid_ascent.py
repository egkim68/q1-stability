"""Step 4. Rapid-ascent flag and its association with retention (chi-square test, Cramer's V).

Rapid ascent: years from Scopus entry to first Q1 at or below the 25th percentile of years-to-Q1 for the
journal's subject area, computed on the full broader sample; a journal tagged with several areas qualifies
if it meets the cutoff in any of them. Years are counted from the journal's first year in the annual-file
panel, and area tags are taken from the file of the first Q1 year.

Usage: python scripts/04_rapid_ascent.py build/state.pkl build/
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

st = pickle.load(open(sys.argv[1], 'rb'))
OUT = sys.argv[2]
p, seqs = st['panel'], st['seqs']
r = pd.read_pickle(os.path.join(OUT, 'journals.pkl'))


def split_areas(s):
    return [a.strip() for a in s.split(';') if a.strip()] if isinstance(s, str) else []


first_year = p.groupby('Sourceid').year.min()
areas_by_year = p.set_index(['Sourceid', 'year']).Areas
r['areas_q1'] = [split_areas(areas_by_year.get((s, int(f)))) for s, f in zip(r.index, r.firstq1)]
r['years_to_q1'] = r.firstq1 - first_year.loc[r.index].values
long = r[['years_to_q1', 'areas_q1']].explode('areas_q1').dropna()
cut = long.groupby('areas_q1').years_to_q1.quantile(.25)
r['rapid'] = [any(v <= cut[a] for a in al) if al else False for v, al in zip(r.years_to_q1, r.areas_q1)]
print(f'rapid ascenders: {r.rapid.sum()} of {len(r)} ({r.rapid.mean() * 100:.1f}%)')

cl = r[r.cat != 'RECENT']
t = pd.crosstab(cl.rapid, cl.in_q1_now)
chi, pval, dof, _ = chi2_contingency(t, correction=False)
print(t)
print(f'retention: rapid {cl[cl.rapid].in_q1_now.mean() * 100:.1f}%, others {cl[~cl.rapid].in_q1_now.mean() * 100:.1f}%')
print(f'chi2({dof}, N = {len(cl)}) = {chi:.1f}, p = {pval:.2g}, Cramer V = {np.sqrt(chi / len(cl)):.3f}')
r.to_pickle(os.path.join(OUT, 'journals.pkl'))
