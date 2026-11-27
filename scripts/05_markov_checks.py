"""Step 5. Checks of the Markov assumptions: persistence by years held in Q1 and by calendar sub-period,
journal-level block bootstrap of the sub-period differences, the first-order check, and the split matrices
(Supplementary Table S1).

Usage: python scripts/05_markov_checks.py build/state.pkl build/markov_checks.json
"""
import pickle, sys, json
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

st = pickle.load(open(sys.argv[1], 'rb'))
P, r, seqs = st['pairs'], st['journals'], st['seqs']
out = {}
q1 = P[P.fr == 1].copy()
q1['stay'] = (q1.to == 1).astype(int)
q1['dbin'] = pd.cut(q1.dur, [0, 3, 6, 100], labels=['1-3', '4-6', '7+'])

# ---- 1a. duration (years since first attainment) and its cross with the calendar split
print('== P11 by years since first attainment')
g = q1.groupby('dbin', observed=True).stay.agg(['mean', 'size'])
for k, row in g.iterrows():
    lo, hi = proportion_confint(row['mean'] * row['size'], row['size'], method='wilson')
    print(f'  {k:4s} P11={row["mean"]:.3f} [{lo:.3f}, {hi:.3f}] n={int(row["size"])}')
out['duration'] = {k: [round(v['mean'], 3), int(v['size'])] for k, v in g.iterrows()}
for split in (2015, 2020):
    q1['period'] = np.where(q1.y0 < split, f'<{split}', f'>={split}')
    t = q1.groupby(['dbin', 'period'], observed=True).stay.agg(['mean', 'size']).unstack()
    print(f'== duration x calendar split {split}'); print(t.round(3).to_string())
    out[f'cross{split}'] = {f'{d}|{per}': [round(t.loc[d, ('mean', per)], 3), int(t.loc[d, ('size', per)])]
                            for d in t.index for per in t['mean'].columns if not np.isnan(t.loc[d, ('mean', per)])}
    # duration-adjusted period difference: weight within-bin differences by pooled bin sizes
    w = q1.groupby('dbin', observed=True).size()
    diff = (t[('mean', f'>={split}')] - t[('mean', f'<{split}')])
    adj = (diff * w).sum() / w[diff.notna()].sum()
    print(f'   raw period gap {q1[q1.y0>=split].stay.mean()-q1[q1.y0<split].stay.mean():.3f}, duration-adjusted {adj:.3f}')
    out[f'adj{split}'] = [round(q1[q1.y0 >= split].stay.mean() - q1[q1.y0 < split].stay.mean(), 3), round(adj, 3)]
# logistic regression with both terms + journal-clustered SE
import statsmodels.formula.api as smf
for split in (2015, 2020):
    q1['post'] = (q1.y0 >= split).astype(int)
    m = smf.logit('stay ~ post + C(dbin)', data=q1).fit(disp=0, cov_type='cluster', cov_kwds={'groups': q1.sid})
    print(f'== logit split {split} (journal-clustered SE)'); print(m.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']].round(4).to_string())
    out[f'logit{split}'] = {k: [round(m.params[k], 3), round(m.bse[k], 3), float(m.pvalues[k])] for k in m.params.index}

# ---- 1b. journal-level block bootstrap
rng = np.random.default_rng(20261124)
B = 2000
by_j = {sid: (g.y0.values, g.stay.values) for sid, g in q1.groupby('sid')}
sids = np.array(list(by_j))
pre_n = {s: {sid: ((v[0] < s).sum(), v[1][v[0] < s].sum()) for sid, v in by_j.items()} for s in (2015, 2020)}
post_n = {s: {sid: ((v[0] >= s).sum(), v[1][v[0] >= s].sum()) for sid, v in by_j.items()} for s in (2015, 2020)}
arr = {}
for s in (2015, 2020):
    arr[s] = np.array([[pre_n[s][sid][0], pre_n[s][sid][1], post_n[s][sid][0], post_n[s][sid][1]] for sid in sids], dtype=float)
res = {s: [] for s in (2015, 2020)}
for b in range(B):
    idx = rng.integers(0, len(sids), len(sids))
    for s in (2015, 2020):
        a = arr[s][idx].sum(0)
        p0, p1 = a[1] / a[0], a[3] / a[2]
        res[s].append((p1 - p0, 1 / (1 - p0), 1 / (1 - p1), 1 / (1 - p1) - 1 / (1 - p0)))
print('== journal-level block bootstrap (B=%d, %d journals)' % (B, len(sids)))
out['boot'] = {}
for s in (2015, 2020):
    x = np.array(res[s]); a = arr[s].sum(0); p0, p1 = a[1] / a[0], a[3] / a[2]
    est = (p1 - p0, 1 / (1 - p0), 1 / (1 - p1), 1 / (1 - p1) - 1 / (1 - p0))
    ci = np.percentile(x, [2.5, 97.5], axis=0)
    names = ['dP11', 'sojourn_pre', 'sojourn_post', 'd_sojourn']
    for i, nm in enumerate(names):
        print(f'  {s} {nm}: {est[i]:.3f} [{ci[0,i]:.3f}, {ci[1,i]:.3f}]')
    out['boot'][s] = {nm: [round(est[i], 3), round(ci[0, i], 3), round(ci[1, i], 3)] for i, nm in enumerate(names)}
    out['boot'][s]['p_le0'] = float((x[:, 0] <= 0).mean())

# ---- 1c. first-order check
tri = []
for sid, row in r[r.cat != 'RECENT'].iterrows():
    s = seqs[sid]
    for t in range(int(row.firstq1) + 2, 2026):
        if t in s and (t - 1) in s and (t - 2) in s and s[t - 1] == 1:
            tri.append((sid, t, int(s[t - 2] == 1), int(s[t] == 1)))
T = pd.DataFrame(tri, columns=['sid', 't', 'q1_tm2', 'stay'])
print('== first-order check: P(Q1_t | Q1_{t-1}, X_{t-2})')
out['first_order'] = {}
for k, g in T.groupby('q1_tm2'):
    lo, hi = proportion_confint(g.stay.sum(), len(g), method='wilson')
    lab = 'Q1,Q1' if k else 'nonQ1,Q1'
    print(f'  {lab}: {g.stay.mean():.3f} [{lo:.3f}, {hi:.3f}] n={len(g)}')
    out['first_order'][lab] = [round(g.stay.mean(), 3), round(lo, 3), round(hi, 3), len(g)]
# bootstrap of the difference, by journal
byj = T.groupby('sid').apply(lambda g: pd.Series({'n1': (g.q1_tm2 == 1).sum(), 's1': g.stay[g.q1_tm2 == 1].sum(),
                                                 'n0': (g.q1_tm2 == 0).sum(), 's0': g.stay[g.q1_tm2 == 0].sum()}), include_groups=False).values.astype(float)
ds = []
for b in range(B):
    a = byj[rng.integers(0, len(byj), len(byj))].sum(0)
    ds.append(a[1] / a[0] - a[3] / a[2])
a = byj.sum(0); dhat = a[1] / a[0] - a[3] / a[2]
ci = np.percentile(ds, [2.5, 97.5])
print(f'  difference {dhat:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (journal bootstrap)')
out['first_order']['diff'] = [round(dhat, 3), round(ci[0], 3), round(ci[1], 3)]
# same check with duration control (logit, clustered)
T = T.merge(r[['firstq1']], left_on='sid', right_index=True)
T['dur'] = T.t - 1 - T.firstq1 + 1
T['dbin'] = pd.cut(T.dur, [0, 3, 6, 100], labels=['1-3', '4-6', '7+'])
m = smf.logit('stay ~ q1_tm2 + C(dbin)', data=T).fit(disp=0, cov_type='cluster', cov_kwds={'groups': T.sid})
print(m.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']].round(4).to_string())
out['first_order']['logit_q1_tm2'] = [round(m.params['q1_tm2'], 3), round(m.bse['q1_tm2'], 3), float(m.pvalues['q1_tm2'])]

# ---- 1d. split matrices
out['matrices'] = {}
for s in (2015, 2020):
    for lab, sub in (('pre', P[P.y0 < s]), ('post', P[P.y0 >= s])):
        M = pd.crosstab(sub.fr, sub.to)
        Mp = M.div(M.sum(axis=1), axis=0)
        out['matrices'][f'{s}_{lab}'] = {'p': Mp.round(3).values.tolist(), 'n': M.sum(axis=1).astype(int).tolist()}
        print(f'== {s} {lab}'); print(Mp.round(3).assign(n=M.sum(axis=1)).to_string())
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=str)
