#!/usr/bin/env python3
"""Generate notebooks/a5_analysis.ipynb (source of truth for the notebook's cells).

Run: uv run python scripts/make_notebook.py && uv run jupyter nbconvert --to notebook --execute \
         --inplace notebooks/a5_analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# ISE-A5: counterexample study of YODA's fault-tolerance claims

This notebook reads **only raw result files** from `results/baseline/` and `results/after/`, plus the
frozen configs. Every number below is recomputed from those files; no result is typed in by hand.
The claims, oracles and thresholds are defined in `docs/protocol_v1.md`, which was frozen at tag
`a5-protocol-frozen`.
""")
code("""
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from a5 import analysis as A
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
base = A.load('configs/baseline.yaml')
after = A.load('configs/after.yaml') if (ROOT / 'results/after/metrics.csv').exists() else None
print('protocol hash file:'); print((ROOT / 'results/protocol_hash.txt').read_text())
""")
md("## 1. Reproduction: training runs and agreement with the published catalogue")
code("""
tr = pd.DataFrame([{k: v for k, v in t.items() if k != 'history'} for t in base['training']])
display(tr)
print('labelled in SoSt:', base['meta']['n_labelled_in_sost'], base['meta']['labelled_by_otype'])
display(base['agreement'])
display(base['agreement'].drop(columns='seed').mean().to_frame('mean over seeds').T)
""")
code("""
fig, ax = plt.subplots(1, len(base['training']), figsize=(4 * len(base['training']), 3), sharey=True)
for a, t in zip(np.atleast_1d(ax), base['training']):
    a.plot(t['history']['val_precision'], label='val PMS precision')
    a.plot(t['history']['precision'], label='train PMS precision', alpha=.6)
    a.set_title(f"seed {t['seed']} ({t['epochs_run']} epochs)"); a.set_xlabel('epoch')
np.atleast_1d(ax)[0].legend(); plt.tight_layout(); plt.show()
""")
md("## 2. Baseline: per-cell results (all grid cells)")
code("""
tb = A.cell_table(base)
cols = ['cell','flip_PMS_cand','flip_PMS_cand_lo','flip_PMS_cand_hi','flip_CBe_cand','flip_CBe_cand_lo',
        'flip_CBe_cand_hi','flip_PMS_test','flip_CBe_test','recall_PMS','d_recall_PMS','d_recall_lo','d_recall_hi',
        'precision_PMS','d_precision_PMS','dN_PMS_rand','dN_CBe_rand','dN_PMS_cand','dN_CBe_cand']
display(tb[cols].round(4))
""")
md("## 3. Decision on C1 (paper-realistic cells, frozen thresholds)")
code("""
db = A.decisions(base, tb)
display(db)
C1_falsified = bool(db.falsifies_C1.any())
print('C1 FALSIFIED' if C1_falsified else 'C1 SURVIVED', '(baseline)')
""")
md("## 4. Flip rate vs fault rate (95% bootstrap CI), PMS and CBe, on published candidates")
code("""
def curves(tab, title):
    faults = [f for f in tab.fault.unique() if f != 'clean']
    fig, axes = plt.subplots(1, len(faults), figsize=(3.6 * len(faults), 3.2), sharey=True)
    for ax, f in zip(axes, faults):
        for k_val, sub in tab[tab.fault == f].groupby(tab.k.fillna(0)):
            sub = pd.concat([tab[tab.fault == 'clean'], sub]).sort_values('rate')
            for c, col in [('PMS', 'C3'), ('CBe', 'C0')]:
                y, lo, hi = sub[f'flip_{c}_cand'], sub[f'flip_{c}_cand_lo'], sub[f'flip_{c}_cand_hi']
                lab = c + (f' k={k_val:g}' if k_val else '')
                ax.errorbar(sub.rate, y, yerr=[y - lo, hi - y], marker='o', capsize=3, label=lab,
                            color=col, alpha=1 if k_val in (0, 2) else .5, ls='-' if k_val in (0, 2) else '--')
        ax.axhline(0.05, color='grey', ls=':', lw=1); ax.set_title(f); ax.set_xlabel('fault rate')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('flip rate (E_cand)'); fig.suptitle(title); plt.tight_layout(); plt.show()
curves(tb, 'Baseline')
""")
md("## 5. Class-transition matrices (rows = clean class, cols = faulted class; E_cand, summed over seeds)")
code("""
names = A.cell_names(base['cfg'])
for rc in base['cfg']['realistic']:
    nm = [n for n, c in zip(names, base['cfg']['cells']) if c['fault'] == rc['fault'] and np.isclose(c['rate'], rc['rate'])][0]
    print(nm); display(pd.DataFrame(A.transitions(base, nm), index=['other', 'PMS', 'CBe'], columns=['->other', '->PMS', '->CBe']))
""")
md("## 6. C2: boundary / malformed-input matrix")
code("""
def c2_table(run):
    c = run['c2']
    rows = [{'case': x['case'], 'kind': x['kind'], 'verdict': x['verdict'], 'outcome': x.get('outcome', '')[:90],
             'detail': x.get('n_altered', x.get('note', ''))} for x in c['cases']]
    b = c['bootgather']
    rows.append({'case': b['case'], 'kind': b['kind'], 'verdict': b['verdict'],
                 'outcome': f"{b.get('n_ids_altered')} of {b.get('n_sources_in')} IDs altered", 'detail': b.get('examples_altered', [])[:2]})
    return pd.DataFrame(rows)
t2b = c2_table(base); display(t2b)
print('original YODA.py as published -> returncode', base['c2']['original_yoda_py_run']['returncode'], ':',
      base['c2']['original_yoda_py_run']['stderr_tail'].strip().splitlines()[-1])
C2_falsified = bool((t2b.verdict == 'FAIL-silent').any())
print('C2 FALSIFIED' if C2_falsified else 'C2 SURVIVED', '(baseline)')
""")
md("## 7. Concrete counterexamples (Gaia DR2 IDs that flipped in realistic cells, first model seed)")
code("""
from a5.data import load_sample
sample = load_sample()
for rc in base['cfg']['realistic']:
    nm = [n for n, c in zip(names, base['cfg']['cells']) if c['fault'] == rc['fault'] and np.isclose(c['rate'], rc['rate'])][0]
    ce, n_flip = A.counterexamples(base, nm, sample, max_rows=10)
    print(f'{nm}: {n_flip} flipped sources in E_cand (seed {base["cfg"]["model_seeds"][0]}); first 10:')
    display(ce)
""")
md("## 8. After the improvement: identical grid, before/after comparison")
code("""
if after is None:
    print('results/after not present yet')
else:
    ta = A.cell_table(after)
    keep = ['flip_PMS_cand', 'flip_CBe_cand', 'd_recall_PMS', 'recall_PMS', 'precision_PMS', 'dN_PMS_rand', 'dN_CBe_rand']
    ba = tb[['cell'] + keep].merge(ta[['cell'] + keep], on='cell', suffixes=('_before', '_after'))
    display(ba.round(4))
    da = A.decisions(after, ta); display(da)
    print('C1 after improvement:', 'FALSIFIED' if da.falsifies_C1.any() else 'SURVIVED')
    t2a = c2_table(after)
    display(t2b[['case', 'verdict']].merge(t2a[['case', 'verdict', 'outcome']], on='case', suffixes=('_before', '_after')))
    print('C2 after improvement:', 'FALSIFIED' if (t2a.verdict == 'FAIL-silent').any() else 'SURVIVED')
    curves(ta, 'After improvement')
""")
md("## 9. Side effects of the improvement")
code("""
if after is not None:
    se = {}
    tb0, ta0 = tb.iloc[0], ta.iloc[0]
    se['clean recall_PMS before/after'] = (tb0.recall_PMS, ta0.recall_PMS)
    se['clean precision_PMS before/after (lower limits)'] = (tb0.precision_PMS, ta0.precision_PMS)
    se['clean N_PMS candidates on E_rand before/after'] = (tb0.N_PMS_rand, ta0.N_PMS_rand)
    se['training+grid wall time [s] before/after'] = (base['meta']['wall_seconds'], after['meta']['wall_seconds'])
    se['training seconds per seed before'] = [t['train_seconds'] for t in base['training']]
    se['training seconds per seed after'] = [t['train_seconds'] for t in after['training']]
    p = ROOT / 'results/after/side_effects.json'
    if p.exists():
        se.update(json.loads(p.read_text()))
    for k, v in se.items(): print(f'{k}: {v}')
""")
md("""
## 10. Verdict, negative results and limits

The verdict cells below print the decision rules evaluated above. The limits apply to every number in
this notebook:

- **Reproduction, not the original model.** No weights or labels were published. Labels are SIMBAD
  `Ae*`/`TT*`/`Be*` stars present in the published table, largely disjoint from the authors' set.
- **Reduced bootstrap.** 3 models instead of the paper's 30. The "other" class is 1/10 of the paper's
  size and training is capped at 300 epochs, so the absolute recall and precision are not comparable
  with 78.8 % / 40.7 %.
- **Synthetic faults.** These are not real mismatched photometry. The F1 noise σ = 1 mag is an
  assumption, and F4 has no literature rate.
- **Gaia DR2-era data.** Inputs from 2018–2020, and SIMBAD as queried on 2026-11-07.
- **Precision is only a lower limit** (paper Appx. B.1).
""")
code("""
print('C1 baseline:', 'FALSIFIED' if C1_falsified else 'SURVIVED')
print(db[['cell', 'falsifies_C1', 'rule_a_PMS', 'rule_a_CBe', 'rule_b_recall']].to_string(index=False))
print('C2 baseline:', 'FALSIFIED' if C2_falsified else 'SURVIVED')
print('C2 FAIL-silent cases:', ', '.join(t2b[t2b.verdict == 'FAIL-silent'].case))
print('C2 FAIL-crash (valid input) cases:', ', '.join(t2b[t2b.verdict == 'FAIL-crash'].case))
fails = list(Path(ROOT / 'results/baseline/failures').glob('*'))
print('crashed grid cells kept in results/baseline/failures:', len(fails))
""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "a5_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
