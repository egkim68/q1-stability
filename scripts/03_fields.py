"""Step 3. Field-level analyses: Kruskal-Wallis tests on entropy (Figure 2), the 27-area dashboard (Figure 3),
and entropy dispersion by ASJC supergroup (Table 4). Classifiable (non-RECENT) journals; each journal is
counted in every area it is tagged with (areas from its earliest annual file).

Usage: python scripts/03_fields.py build/
"""
import os
import sys
import pandas as pd
from scipy.stats import kruskal, spearmanr

OUT = sys.argv[1]
r = pd.read_pickle(os.path.join(OUT, 'journals.pkl'))
cl = r[r.cat != 'RECENT'].copy()
cl['areas'] = cl.areas_first.map(lambda s: [a.strip() for a in s.split(';') if a.strip()] if isinstance(s, str) else [])
L = cl.explode('areas').dropna(subset=['areas'])
L['dist'] = L.mean_quartile - 1
L['sustained'] = L.cat == 'SUSTAINED'

# Figure 2: entropy across the eight largest areas
top8 = L.areas.value_counts().index[:8]
h = kruskal(*[L[L.areas == a].entropy for a in top8])
print(f'Kruskal-Wallis, 8 largest areas: H(7) = {h.statistic:.1f}, p = {h.pvalue:.2g}')

# single-tagged robustness checks
single = cl[cl.areas.str.len() == 1].copy()
single['a'] = single.areas.str[0]
s8 = single[single.a.isin(top8)]
h = kruskal(*[s8[s8.a == a].entropy for a in top8])
print(f'single-tagged ({len(single)}), 8 areas: H(7) = {h.statistic:.1f}, n = {len(s8)}, eps2 = {h.statistic / (len(s8) - 1):.2f}')
counts = single.a.value_counts()
big = counts[counts >= 10].index
s21 = single[single.a.isin(big)]
h = kruskal(*[s21[s21.a == a].entropy for a in big])
print(f'single-tagged, {len(big)} areas with n >= 10: H({len(big) - 1}) = {h.statistic:.1f}, n = {len(s21)}, eps2 = {h.statistic / (len(s21) - 1):.2f}')

# Figure 3: dashboard
g = L.groupby('areas')
dash = pd.DataFrame({'n': g.size(), 'distance_from_peak': g.dist.mean(), 'downgrade_probability': 1 - g.sustained.mean(),
                     'entropy': g.entropy.mean(), 'drawdown': g.drawdown.mean(), 'turnover': g.transitions.mean(),
                     'window_years': g.window.mean()})
dash['turnover_per_year'] = dash.turnover / dash.window_years
print('\n== Figure 3 dashboard (sorted by entropy)')
print(dash.sort_values('entropy').round(2).to_string())
rho = spearmanr(dash.turnover, dash.turnover_per_year).statistic
print(f'Spearman rank correlation, turnover vs per-year turnover: {rho:.3f}')
dash.to_csv(os.path.join(OUT, 'figure3_dashboard.csv'))

# Table 4: supergroups (Kim & Jeong, 2023; Multidisciplinary in Social Sciences)
SUPER = {
    'Life Sciences': ['Agricultural and Biological Sciences', 'Biochemistry, Genetics and Molecular Biology',
                      'Immunology and Microbiology', 'Neuroscience', 'Pharmacology, Toxicology and Pharmaceutics'],
    'Health Sciences': ['Medicine', 'Nursing', 'Veterinary', 'Dentistry', 'Health Professions'],
    'Physical Sciences': ['Chemical Engineering', 'Chemistry', 'Computer Science', 'Earth and Planetary Sciences', 'Energy',
                          'Engineering', 'Environmental Science', 'Materials Science', 'Mathematics', 'Physics and Astronomy'],
    'Social Sciences': ['Arts and Humanities', 'Business, Management and Accounting', 'Decision Sciences',
                        'Economics, Econometrics and Finance', 'Psychology', 'Social Sciences', 'Multidisciplinary'],
}
amap = {a: k for k, v in SUPER.items() for a in v}
L['supergroup'] = L.areas.map(amap)
rows = []
for k, d in L.groupby('supergroup'):
    q = d.entropy.quantile([.25, .5, .75])
    rows.append(dict(supergroup=k, n=len(d), median=q[.5], sd=d.entropy.std(), iqr=q[.75] - q[.25],
                     pct_near_zero=(d.entropy < 0.1).mean() * 100))
t4 = pd.DataFrame(rows).set_index('supergroup')
print('\n== Table 4: entropy dispersion by supergroup')
print(t4.round(3).to_string())
t4.to_csv(os.path.join(OUT, 'table4_supergroups.csv'))
