"""Step 2. Journal-level stability metrics (Table 2) and the pooled quartile transition matrix (Table 3).

Usage: python scripts/02_metrics_markov.py build/state.pkl build/

Metrics, computed over each journal's observed post-attainment quartiles:
  mean quartile level (Q1 = 1 ... Q4 = 4), transition count (changes between consecutive observed years),
  longest Q1 run, Shannon entropy in bits, maximum drawdown (worst quartile minus one).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

st = pickle.load(open(sys.argv[1], 'rb'))
OUT = sys.argv[2]
r, seqs, P = st['journals'], st['seqs'], st['pairs']


def metrics(sid, f):
    s = seqs[sid]
    qs = np.array([s[y] for y in sorted(s) if y >= f])
    counts = np.bincount(qs, minlength=5)[1:]
    p = counts[counts > 0] / counts.sum()
    return dict(mean_quartile=qs.mean(), transitions=int((qs[1:] != qs[:-1]).sum()),
                entropy=float(-(p * np.log2(p)).sum()), drawdown=int(qs.max() - 1), window=2025 - f + 1)


M = pd.DataFrame([metrics(s, f) for s, f in zip(r.index, r.firstq1)], index=r.index)
r = r.join(M)
r.to_pickle(os.path.join(OUT, 'journals.pkl'))

order = ['SUSTAINED', 'RECOVERING', 'FLUCTUATING', 'DECLINING']
t2 = r[r.cat != 'RECENT'].groupby('cat')[['mean_quartile', 'transitions', 'longest_q1_run', 'entropy', 'drawdown']].mean().loc[order]
print('== Table 2: metrics by QTT category (classifiable journals)')
print(t2.round(2).to_string())

M3 = pd.crosstab(P.fr, P.to)
Pm = M3.div(M3.sum(axis=1), axis=0)
print('\n== Table 3: transition matrix P(row quartile in t-1 -> column quartile in t)')
print(Pm.round(3).assign(n=M3.sum(axis=1)).to_string())
for k in range(1, 5):
    lo, hi = proportion_confint(M3.loc[k, k], M3.loc[k].sum(), method='wilson')
    print(f'  persistence Q{k}: {Pm.loc[k, k]:.3f}  Wilson 95% CI [{lo:.3f}, {hi:.3f}]')
p11 = round(Pm.loc[1, 1], 3)
print(f'expected Q1 sojourn = 1 / (1 - {p11:.3f}) = {1 / (1 - p11):.1f} years')
t2.to_csv(os.path.join(OUT, 'table2_metrics_by_qtt.csv'))
Pm.assign(n=M3.sum(axis=1)).to_csv(os.path.join(OUT, 'table3_transition_matrix.csv'))
