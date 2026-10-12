"""Generate every table and figure of the manuscript directly from the verified result files (no hand-copied numbers).
Output: Paper/Manuscript_Acquisition_Design_2026_10_12/{tables,figures,numbers.json}. Reads results only; modifies nothing else."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
S = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
T3 = ROOT / 'idea1_2026/stepup_third_task_v1'
OUT = ROOT / 'Paper/Manuscript_Acquisition_Design_2026_10_12'
(OUT / 'tables').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
NUM = {}

BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6b6a66'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': SURF, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})


def f3(x):
    return f'{x:.3f}'


def ci(m, lo, hi, d=3):
    return f'{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def _pt(x):
    v = round(100 * x, 1)
    return f'{0.0:+.1f}' if v == 0 else f'{v:+.1f}'


def pp(m, lo, hi):
    return f'{_pt(m)} [{_pt(lo)}, {_pt(hi)}]'


def md_table(df, align=None):
    cols = list(df.columns); lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines) + '\n'


def save(name, df):
    (OUT / 'tables' / f'{name}.md').write_text(md_table(df), encoding='utf-8'); df.to_csv(OUT / 'tables' / f'{name}.csv', index=False)


# ------------------------------------------------------------------ surfaces
SURF_FILES = {'FPI': S / 'grid_v1', 'GaitRec': S / 'gaitrec_grid_v1', 'StepUP_sex': T3 / 'surface_L1_v1', 'StepUP_bmi': T3 / 'surface_L2_v1'}
surfaces = {k: pd.read_csv(v / 'learning_surface.csv') for k, v in SURF_FILES.items()}
contrast_files = {'FPI': S / 'grid_v1/prespecified_contrasts.csv', 'GaitRec': S / 'gaitrec_grid_v1/prespecified_contrasts.csv',
                  'StepUP_sex': T3 / 'surface_L1_v1/prespecified_contrasts.csv', 'StepUP_bmi': T3 / 'surface_L2_v1/prespecified_contrasts.csv'}


def surface_table(name):
    s = surfaces[name]; ks = sorted(s.k.unique()); rows = []
    for n in sorted(s.n.unique()):
        row = {'People (n)': n}
        for k in ks:
            r = s[(s.n == n) & (s.k == k)].iloc[0]; row[f'k = {k}'] = ci(r.balanced_accuracy, r.ba_ci_low, r.ba_ci_high)
        rows.append(row)
    return pd.DataFrame(rows)


for nm in surfaces:
    save(f'surface_{nm}', surface_table(nm))
    s = surfaces[nm]; NUM[f'surface_{nm}'] = {f'{int(r.n)}x{int(r.k)}': round(float(r.balanced_accuracy), 4) for _, r in s.iterrows()}

# ------------------------------------------------------------------ equal-budget contrasts
rows = []; forest = []
LABELS = {'FPI': 'FPI posture classes (70 men)', 'GaitRec': 'GaitRec diagnosis classes (1,301 patients)', 'StepUP_sex': 'StepUP sex (150)', 'StepUP_bmi': 'StepUP BMI ≥ 25 (150)'}
for nm, f in contrast_files.items():
    c = pd.read_csv(f)
    if nm == 'FPI':
        sens = pd.read_csv(S / 'sensitivity_v1/tuned_contrasts.csv'); fixed_ll = sens[(sens.regime == 'fixed_grid_v1') & (sens.metric == 'log_loss')].iloc[0]
        for _, r in c.iterrows():
            ll = ci(fixed_ll['mean'], fixed_ll.ci_low, fixed_ll.ci_high) if r.contrast.startswith('primary') else 'not computed'
            rows.append({'Task': LABELS[nm], 'Contrast (people x trials)': r.contrast.replace('primary_', '').replace('secondary_', '').replace('_minus_', ' minus '),
                         'Recordings (A / B)': f'{int(r.recordings_A)} / {int(r.recordings_B)}', 'Role': 'primary, equal budget' if r.contrast.startswith('primary') else 'near budget (168 vs 175)',
                         'BA difference, points [95% CI]': pp(r.mean_difference, r.ci_low, r.ci_high), 'Pairs favouring A': f'{int(round(r.fraction_pairs_positive*60))}/60', 'Log-loss difference [95% CI]': ll})
            forest.append((LABELS[nm], r.contrast.replace('primary_', '').replace('secondary_', '').replace('_minus_', ' − '), r.mean_difference, r.ci_low, r.ci_high, r.contrast.startswith('primary')))
        NUM['FPI_primary'] = {'mean': float(c.iloc[0].mean_difference), 'lo': float(c.iloc[0].ci_low), 'hi': float(c.iloc[0].ci_high)}
        NUM['FPI_secondary'] = {'mean': float(c.iloc[1].mean_difference), 'lo': float(c.iloc[1].ci_low), 'hi': float(c.iloc[1].ci_high)}
        NUM['FPI_primary_logloss'] = {'mean': float(fixed_ll['mean']), 'lo': float(fixed_ll.ci_low), 'hi': float(fixed_ll.ci_high)}
        continue
    for ctr in c.contrast.unique():
        b = c[(c.contrast == ctr) & (c.metric == 'balanced_accuracy')].iloc[0]; l = c[(c.contrast == ctr) & (c.metric == 'log_loss')].iloc[0]
        exact = int(b.recordings_A) == int(b.recordings_B)
        rows.append({'Task': LABELS[nm], 'Contrast (people x trials)': ctr.replace('primary_', '').replace('secondary_', '').replace('_minus_', ' minus '), 'Recordings (A / B)': f'{int(b.recordings_A)} / {int(b.recordings_B)}',
                     'Role': ('primary, ' if ctr.startswith('primary') else 'secondary, ') + 'equal budget', 'BA difference, points [95% CI]': pp(b['mean'], b.ci_low, b.ci_high),
                     'Pairs favouring A': f'{int(b.pairs_positive)}/{int(b.paired_pairs)}', 'Log-loss difference [95% CI]': ci(l['mean'], l.ci_low, l.ci_high)})
        forest.append((LABELS[nm], ctr.replace('primary_', '').replace('secondary_', '').replace('_minus_', ' − '), b['mean'], b.ci_low, b.ci_high, ctr.startswith('primary')))
        NUM[f'{nm}_{ctr}'] = {'mean': float(b['mean']), 'lo': float(b.ci_low), 'hi': float(b.ci_high), 'll_mean': float(l['mean']), 'll_lo': float(l.ci_low), 'll_hi': float(l.ci_high)}
save('equal_budget_contrasts', pd.DataFrame(rows))

# ------------------------------------------------------------------ variability
v = pd.read_csv(S / 'grid_v1/contrast_variability_decomposition_v2.csv')
vr = []
for _, r in v.iterrows():
    vr.append({'Dataset': 'FPI', 'Contrast': r.contrast.replace('primary_', '').replace('secondary_', '').replace('_minus_', ' minus '), 'Mean (points)': f'{100*r["mean"]:+.1f}',
               'Within-repeat SD over 20 draws (points; repeats 1/2/3)': f'{100*r.within_repeat_sd_r0:.1f} / {100*r.within_repeat_sd_r1:.1f} / {100*r.within_repeat_sd_r2:.1f}',
               'SD of the three repeat means (points)': f'{100*r.sd_of_repeat_means:.2f}', 'Pairs favouring A per repeat': r.pairs_positive_per_repeat.replace('/', ' / ')})
save('variability_fpi_contrasts', pd.DataFrame(vr))
for nm in ['GaitRec', 'StepUP_sex', 'StepUP_bmi']:
    pass
vrows = []
for nm, f in contrast_files.items():
    c = pd.read_csv(f)
    if nm == 'FPI':
        r = v.iloc[0]; vrows.append({'Dataset': LABELS[nm], 'Primary contrast': '35x3 minus 21x5', 'Mean (points)': f'{100*r["mean"]:+.1f}', 'Within-repeat SD over draws (points)': f'{100*r.within_repeat_sd_mean:.1f}', 'SD of the three repeat means (points)': f'{100*r.sd_of_repeat_means:.2f}'})
    else:
        r = c[(c.metric == 'balanced_accuracy') & (c.contrast.str.startswith('primary'))].iloc[0]
        vrows.append({'Dataset': LABELS[nm], 'Primary contrast': r.contrast.replace('primary_', '').replace('_minus_', ' minus '), 'Mean (points)': f'{100*r["mean"]:+.1f}', 'Within-repeat SD over draws (points)': f'{100*r.within_repeat_sd_mean:.1f}', 'SD of the three repeat means (points)': f'{100*r.sd_of_repeat_means:.2f}'})
save('variability_primary_contrasts', pd.DataFrame(vrows))
NUM['FPI_primary_var'] = {'within': float(v.iloc[0].within_repeat_sd_mean), 'between': float(v.iloc[0].sd_of_repeat_means)}
sv = pd.read_csv(S / 'grid_v1/variability_decomposition_v2.csv'); r56 = sv[(sv.n == 56) & (sv.k == 5)].iloc[0]; r21 = sv[(sv.n == 21) & (sv.k == 3)].iloc[0]
NUM['FPI_var_56x5_within'] = float(r56.within_repeat_sd_mean); NUM['FPI_var_21x3_within'] = float(r21.within_repeat_sd_mean)

# ------------------------------------------------------------------ finite differences summary
fd_rows = []
for nm, d in SURF_FILES.items():
    f = pd.read_csv(d / 'finite_differences.csv')
    for ax in ['n', 'k']:
        g = f[f.axis == ax]; pos = int((g.mean_difference > 0).sum()); sig = int(((g.ci_low > 0)).sum()); neg_sig = int((g.ci_high < 0).sum())
        fd_rows.append({'Task': LABELS[nm], 'Axis': 'adding people' if ax == 'n' else 'adding trials per person', 'Steps': len(g), 'Positive point estimates': pos, 'CI above zero': sig, 'CI below zero': neg_sig})
        NUM[f'fd_{nm}_{ax}'] = {'steps': len(g), 'pos': pos, 'sig': sig, 'neg_sig': neg_sig}
save('finite_difference_summary', pd.DataFrame(fd_rows))
# k 3 -> 5 steps
k35 = []
for nm, d in SURF_FILES.items():
    f = pd.read_csv(d / 'finite_differences.csv'); g = f[(f.axis == 'k') & (f['from'].str.endswith('x3')) & (f['to'].str.endswith('x5'))]
    for _, r in g.iterrows():
        k35.append({'Task': LABELS[nm], 'People (n)': r['from'].split('x')[0], 'Gain from 3 to 5 trials, points [95% CI]': pp(r.mean_difference, r.ci_low, r.ci_high)})
save('gain_3_to_5_trials', pd.DataFrame(k35))
# StepUP sex k steps 5->10->20 at n=120
f = pd.read_csv(T3 / 'surface_L1_v1/finite_differences.csv'); a = f[f['from'] == '120x5'].iloc[0]; b = f[f['from'] == '120x10'].iloc[0]
NUM['StepUP_sex_120_5_10'] = {'mean': float(a.mean_difference), 'lo': float(a.ci_low), 'hi': float(a.ci_high)}; NUM['StepUP_sex_120_10_20'] = {'mean': float(b.mean_difference), 'lo': float(b.ci_low), 'hi': float(b.ci_high)}
f2 = pd.read_csv(T3 / 'surface_L2_v1/finite_differences.csv'); c2 = f2[f2['from'] == '120x10'].iloc[0]; NUM['StepUP_bmi_120_10_20'] = {'mean': float(c2.mean_difference), 'lo': float(c2.ci_low), 'hi': float(c2.ci_high)}

# ------------------------------------------------------------------ sensitivity (inner tuning)
sens = pd.read_csv(S / 'sensitivity_v1/tuned_contrasts.csv'); cells = pd.read_csv(S / 'sensitivity_v1/tuned_vs_fixed_by_cell.csv'); dist = pd.read_csv(S / 'sensitivity_v1/selected_C_distribution.csv')
st1 = pd.read_csv(T3 / 'sensitivity_L1_v1/contrast_tuned_vs_fixed.csv'); stc = pd.read_csv(T3 / 'sensitivity_L1_v1/cells_tuned_vs_fixed.csv'); sdist = pd.read_csv(T3 / 'sensitivity_L1_v1/selected_C_distribution.csv')
rows = []
for _, r in sens.iterrows():
    rows.append({'Dataset': 'FPI (35x3 minus 21x5)', 'C regime': 'inner-tuned' if r.regime == 'tuned' else 'fixed 0.1', 'Metric': r.metric.replace('_', ' '), 'Difference [95% CI]': ci(r['mean'], r.ci_low, r.ci_high)})
for _, r in st1.iterrows():
    rows.append({'Dataset': 'StepUP sex (120x5 minus 60x10)', 'C regime': 'inner-tuned' if r.regime == 'tuned' else 'fixed 0.1', 'Metric': r.metric.replace('_', ' '), 'Difference [95% CI]': ci(r['mean'], r.ci_low, r.ci_high)})
save('sensitivity_contrasts', pd.DataFrame(rows))
d_all = dist.groupby('n').agg({'0.1': 'sum', '1.0': 'sum', '10.0': 'sum'}); sd_all = sdist.set_index('n')[['0.1', '1.0', '10.0']]
crow = []
for lab, d, cell in [('FPI', d_all, {35: '35x3', 21: '21x5'}), ('StepUP sex', sd_all, {120: '120x5', 60: '60x10'})]:
    for n, r in d.iterrows():
        tot = int(r.sum()); crow.append({'Dataset': lab, 'Cell': cell[n], 'Inner fits selecting C = 0.1': f'{int(r["0.1"])} ({100*r["0.1"]/tot:.0f}%)', 'C = 1': int(r['1.0']), 'C = 10': int(r['10.0']), 'Acquired sets': tot})
save('selected_C', pd.DataFrame(crow))
NUM['sens_FPI_tuned'] = {'mean': float(sens[(sens.regime == 'tuned') & (sens.metric == 'balanced_accuracy')].iloc[0]['mean']), 'lo': float(sens[(sens.regime == 'tuned') & (sens.metric == 'balanced_accuracy')].iloc[0].ci_low),
                         'hi': float(sens[(sens.regime == 'tuned') & (sens.metric == 'balanced_accuracy')].iloc[0].ci_high)}
NUM['sens_FPI_C01_share'] = {str(n): float(r['0.1'] / r.sum()) for n, r in d_all.iterrows()}
tcell = cells[cells.metric == 'log_loss']; NUM['sens_FPI_ll_worse'] = {r.cell: [float(r.tuned_minus_fixed_mean), float(r.tuned_minus_fixed_ci_low), float(r.tuned_minus_fixed_ci_high)] for _, r in tcell.iterrows()}

# ------------------------------------------------------------------ policies
POLICY_SRC = [('FPI', 'FPI', S / 'policy_grid_v1/FPI_policy_summary.csv', S / 'policy_grid_v1/FPI_acquired_composition.csv', [105, 70], 'A4'),
              ('GaitRec', 'GaitRec', S / 'policy_grid_v1/GaitRec_policy_summary.csv', S / 'policy_grid_v1/GaitRec_acquired_composition.csv', [1500, 3000], 'A4'),
              ('StepUP sex', 'L1', T3 / 'policy_L1_v1/policy_summary.csv', T3 / 'policy_L1_v1/acquired_composition.csv', [300, 600], 'A5'),
              ('StepUP BMI', 'L2', T3 / 'policy_L2_v1/policy_summary.csv', T3 / 'policy_L2_v1/acquired_composition.csv', [300, 600], 'A5'),
              ('StepUP sex', 'L1', T3 / 'policy_a52_L1_v1/policy_summary.csv', T3 / 'policy_a52_L1_v1/acquired_composition.csv', [150, 100], 'A5.2'),
              ('StepUP BMI', 'L2', T3 / 'policy_a52_L2_v1/policy_summary.csv', T3 / 'policy_a52_L2_v1/acquired_composition.csv', [150, 100], 'A5.2')]
prow = []; pforest = []; pbudget_info = {}; SHARES = []
for label, _, sf, cf, budgets, stage in POLICY_SRC:
    s = pd.read_csv(sf); comp = pd.read_csv(cf); dec_file = Path(str(sf).replace('policy_summary.csv', 'policy_decisions.csv').replace('FPI_policy_decisions.csv', 'FPI_policy_decisions.csv'))
    if 'policy_grid_v1' in str(sf):
        dec_file = sf.parent / (sf.name.split('_')[0] + '_policy_decisions.csv')
    if 'task' in comp.columns:
        pass
    for B in budgets:
        ba = s[(s.metric == 'balanced_accuracy') & (s.budget == B)].drop_duplicates('policy').set_index('policy')
        extra = [i for i in ba.index if i.startswith('MARGIN_minus_') and i not in ('MARGIN_minus_CONV5', 'MARGIN_minus_RANDOM')]
        best = extra[0].replace('MARGIN_minus_', '').replace('_ORACLEFIX', '') if extra else 'CONV5'  # post-hoc best fixed design exactly as selected inside the run
        fixed = {p: ba.loc[p, 'mean'] for p in ba.index if (p.startswith('FIX') or p == 'CONV5') and not p.startswith('MARGIN')}
        saturated = float(comp[(comp.budget == B) & (comp.policy == 'MARGIN')].mean_people.iloc[0]) >= 0.98 * float(comp[(comp.budget == B) & (comp.policy == 'FIX1')].mean_people.iloc[0]) and B in (3000, 600)
        def d(ref):
            key = f'MARGIN_minus_{ref}' if f'MARGIN_minus_{ref}' in ba.index else f'MARGIN_minus_{ref}_ORACLEFIX'
            r = ba.loc[key]; return r['mean'], r.ci_low, r.ci_high
        dC, dR, dB = d('CONV5'), d('RANDOM'), d(best)
        dec = pd.read_csv(dec_file); dm = dec[(dec.policy == 'MARGIN') & (dec.budget == B)].drop_duplicates(['repeat', 'fold', 'seed', 'budget', 'step'])
        share_people = float((dm.action == 'people').mean()); undef = float(dm.undefined_estimate.astype(bool).mean()) if 'undefined_estimate' in dm else 0.0
        cm = comp[(comp.budget == B) & (comp.policy == 'MARGIN')].iloc[0]; cc = comp[(comp.budget == B) & (comp.policy == 'CONV5')].iloc[0]
        prow.append({'Task': label, 'Stage': stage, 'Budget': B, 'Pool-saturated': 'yes' if saturated else 'no', 'CONV5': f3(ba.loc['CONV5', 'mean']), 'Best fixed (post hoc)': f'{best} {f3(fixed[best])}', 'RANDOM': f3(ba.loc['RANDOM', 'mean']),
                     'MARGIN': f3(ba.loc['MARGIN', 'mean']), 'MARGIN − CONV5, points [95% CI]': pp(*dC), 'MARGIN − RANDOM, points [95% CI]': pp(*dR), 'MARGIN − best fixed, points [95% CI]': pp(*dB),
                     'MARGIN people x trials': f'{cm.mean_people:.0f} x {cm.mean_trials_per_person:.2f}', 'CONV5 people x trials': f'{cc.mean_people:.0f} x {cc.mean_trials_per_person:.1f}', 'MARGIN batches to people': f'{100*share_people:.0f}%', 'Undefined internal estimates': f'{100*undef:.1f}%'})
        SHARES.append((label, stage, B, share_people, undef, saturated))
        pforest.append((f'{label}, B = {B} ({stage})', dC, dR, dB, saturated, best))
        NUM[f'policy_{label}_{stage}_{B}'] = {'conv5': [float(x) for x in dC], 'random': [float(x) for x in dR], 'best': [float(x) for x in dB], 'best_name': best, 'saturated': bool(saturated)}
pdf_ = pd.DataFrame(prow)
save('policy_results', pdf_)
save('policy_ba', pdf_[['Task', 'Stage', 'Budget', 'Pool-saturated', 'CONV5', 'Best fixed (post hoc)', 'RANDOM', 'MARGIN']])
save('policy_contrasts', pdf_[['Task', 'Budget', 'MARGIN − CONV5, points [95% CI]', 'MARGIN − RANDOM, points [95% CI]', 'MARGIN − best fixed, points [95% CI]', 'MARGIN people x trials', 'CONV5 people x trials', 'MARGIN batches to people', 'Undefined internal estimates']])

ns_ = [x for x in SHARES if not x[5]]
NUM['policy_share_people_range'] = [min(x[3] for x in ns_), max(x[3] for x in ns_)]
NUM['policy_undefined'] = {f'{x[0]}_{x[1]}_{x[2]}': x[4] for x in SHARES}

# ------------------------------------------------------------------ inference-side budget (FPI, supplement)
fm = pd.read_csv(ROOT / 'idea1_2026/fixed_model_trial_budget_audit_v1/fixed_model_performance_ci.csv'); fc = pd.read_csv(ROOT / 'idea1_2026/fixed_model_trial_budget_audit_v1/fixed_model_paired_contrasts.csv')
fi = pd.read_csv(ROOT / 'idea1_2026/fixed_model_trial_budget_audit_v1/fixed_model_choice_instability.csv')
rows = []
for k in range(1, 6):
    r = fm[(fm.method == 'Median') & (fm.k == k)].iloc[0]; i = fi[(fi.method == 'Median') & (fi.k == k)].iloc[0]
    rows.append({'Trials available at assessment (k)': k, 'Median-pooled BA [95% CI]': ci(r.expected_ba, r.conditional_ci_low, r.conditional_ci_high), 'Person-repeat assessments whose class depends on which k trials were recorded': f'{100*i.fraction_person_repeat_any_disagreement:.1f}%'})
save('inference_budget_fpi', pd.DataFrame(rows))
m51 = fc[(fc.method == 'Median') & (fc.inference_k_candidate == 5) & (fc.inference_k_baseline == 1)].iloc[0]; m35 = fc[(fc.method == 'Median') & (fc.inference_k_candidate == 3) & (fc.inference_k_baseline == 5)].iloc[0]
NUM['inference_median_5_1'] = [float(m51.fixed_model_ba_delta), float(m51.conditional_ci_low), float(m51.conditional_ci_high)]; NUM['inference_median_3_5'] = [float(m35.fixed_model_ba_delta), float(m35.conditional_ci_low), float(m35.conditional_ci_high)]
NUM['inference_k3_disagree'] = float(fi[(fi.method == 'Median') & (fi.k == 3)].iloc[0].fraction_person_repeat_any_disagreement)

# ------------------------------------------------------------------ verification / execution log
LOG = [('FPI mechanical pilot', S / 'pilot_v1'), ('FPI learning surface (A1)', S / 'grid_v1'), ('FPI inner-tuning sensitivity (A2)', S / 'sensitivity_v1'), ('GaitRec mechanical pilot', S / 'gaitrec_pilot_v1'),
       ('GaitRec learning surface (A3)', S / 'gaitrec_grid_v1'), ('FPI + GaitRec sequential rule (A4)', S / 'policy_grid_v1'), ('StepUP pilot (A5)', T3 / 'pilot_v1'), ('StepUP surface, sex (A5)', T3 / 'surface_L1_v1'),
       ('StepUP surface, BMI (A5)', T3 / 'surface_L2_v1'), ('StepUP sensitivity, sex (A5)', T3 / 'sensitivity_L1_v1'), ('StepUP sequential rule, sex (A5)', T3 / 'policy_L1_v1'), ('StepUP sequential rule, BMI (A5)', T3 / 'policy_L2_v1'),
       ('StepUP small-budget rule, sex (A5.2)', T3 / 'policy_a52_L1_v1'), ('StepUP small-budget rule, BMI (A5.2)', T3 / 'policy_a52_L2_v1')]
lrows = []; tot = {'primary': 0, 'final': 0, 'internal': 0, 'inner': 0, 'refit': 0, 'control': 0}
for name, d in LOG:
    pr = json.loads((d / 'progress.json').read_text()); vr_ = json.loads((d / 'independent_verification.json').read_text())
    prim = pr.get('primary_fits', 0) + pr.get('pilot_logistic_fits', 0); fin = pr.get('final_fits', 0); itn = pr.get('internal_fits', 0); inn = pr.get('inner_fits', 0); ref = pr.get('refits', 0)
    ctl = pr.get('exclusion_control_fits', pr.get('exclusion_proof_duplicate_fits', 0))
    if name == 'FPI inner-tuning sensitivity (A2)':
        pass
    lrows.append({'Stage': name, 'Model fits': f'{prim + fin:,}' if prim + fin else '-', 'Policy-internal fits': f'{itn:,}' if itn else '-', 'Inner-tuning fits': f'{inn:,}' if inn else '-', 'Refits': f'{ref:,}' if ref else '-',
                  'Exclusion-control fits': f'{ctl:,}' if ctl else '-', 'Independent verifier': vr_['status'].replace('PASS_LISTED_CHECKS_ONLY', 'pass (listed checks)').replace('PASS_PILOT_MECHANICAL_CHECKS', 'pass (listed checks)')})
    tot['primary'] += prim; tot['final'] += fin; tot['internal'] += itn; tot['inner'] += inn; tot['refit'] += ref; tot['control'] += ctl
save('execution_log', pd.DataFrame(lrows)); NUM['totals'] = tot

# ------------------------------------------------------------------ protocol hashes
hrows = []
for f in sorted(S.glob('frozen_protocol_A*.json')) + [S / 'frozen_protocol.json']:
    if 'before' in f.name or 'attempt' in f.name or 'DRAFT' in f.name:
        continue
    d = json.loads(f.read_text(encoding='utf-8'))
    for k_, v_ in d.items():
        if isinstance(v_, str) and len(v_) == 64 and ('addendum' in k_ or 'protocol_md' in k_ or k_ == 'script_sha256' or k_ == 'grid_script_sha256'):
            hrows.append({'Protocol file': f.name, 'Field': k_, 'SHA-256 (recorded before execution)': v_})
save('protocol_hashes', pd.DataFrame(hrows))

# ------------------------------------------------------------------ figures
SEQ = ['#bcd6f2', '#86b3e8', '#4f8fdc', '#2a78d6', '#1b4f93']


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), sharey=False)
    for ax, (nm, title) in zip(axes.ravel(), [('FPI', 'A  FPI posture classes (70 men)'), ('GaitRec', 'B  GaitRec diagnosis classes (1,301)'), ('StepUP_sex', 'C  StepUP sex (150)'), ('StepUP_bmi', 'D  StepUP BMI ≥ 25 (150)')]):
        s = surfaces[nm]; ks = sorted(s.k.unique()); cols = SEQ if len(ks) == 5 else [SEQ[1], SEQ[3], SEQ[4]]
        ends = []
        for k, c in zip(ks, cols):
            g = s[s.k == k].sort_values('n'); ax.plot(g.n, g.balanced_accuracy, '-o', color=c, ms=4, lw=1.8, label=f'{k}'); ends.append([k, float(g.balanced_accuracy.iloc[-1]), float(g.n.iloc[-1])])
        ends.sort(key=lambda e: e[1]); gap = 0.035 * (s.balanced_accuracy.max() - s.balanced_accuracy.min())
        for i in range(1, len(ends)):
            if ends[i][1] - ends[i - 1][1] < gap:
                ends[i].append(ends[i - 1][1] + gap); ends[i][1] = ends[i - 1][1] + gap
        for k, y_, x_ in [(e[0], e[1], e[2]) for e in ends]:
            ax.text(x_ * 1.02, y_, f'k={k}', fontsize=7, color=INK2, va='center')
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold'); ax.set_xlabel('Training people (n)'); ax.set_ylabel('Balanced accuracy'); ax.set_xlim(right=s.n.max() * 1.18)
    fig.tight_layout(); fig.savefig(OUT / 'figures/fig1_learning_surfaces.png', dpi=200); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(7.4, 5.4)); y = 0; yt = []; yl = []
    for lab in [LABELS['FPI'], LABELS['GaitRec'], LABELS['StepUP_sex'], LABELS['StepUP_bmi']]:
        rows_ = [f for f in forest if f[0] == lab]
        ax.text(0.0, y, lab, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8.5, fontweight='bold', x=-0.62) if False else ax.text(-0.62, y, lab, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8.5, fontweight='bold', clip_on=False); y += 1
        for _, ctr, m, lo, hi, primary in rows_:
            ax.plot([100 * lo, 100 * hi], [y, y], color=BLUE, lw=2.2 if primary else 1.4, solid_capstyle='round')
            ax.plot(100 * m, y, 'o', color=BLUE if primary else SURF, mec=BLUE, ms=6 if primary else 5, zorder=3)
            yt.append(y); yl.append(ctr); y += 1
        y += 0.6
    ax.axvline(0, color=INK2, lw=0.9); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_ylim(y - 0.4, -0.9)
    ax.set_xlabel('More people minus more trials: BA difference (points, 95% CI)'); ax.grid(axis='y', visible=False)
    fig.subplots_adjust(left=0.40, right=0.97, bottom=0.12, top=0.98); fig.savefig(OUT / 'figures/fig2_equal_budget_contrasts.png', dpi=200); plt.close(fig)


def fig3():
    rows_ = [p for p in pforest if not p[4]]; fig, ax = plt.subplots(figsize=(7.4, 5.6)); y = 0; yt = []; yl = []
    style = [('MARGIN − CONV5', ORANGE, 'o', 1), ('MARGIN − RANDOM', GRAY, 's', 2), ('MARGIN − best fixed (post hoc)', AQUA, 'D', 3)]
    for lab, dC, dR, dB, sat, best in rows_:
        for (nm, col, mk, idx), val in zip(style, [dC, dR, dB]):
            ax.plot([100 * val[1], 100 * val[2]], [y, y], color=col, lw=1.6); ax.plot(100 * val[0], y, mk, color=col, ms=5.5, zorder=3)
            y += 0.28
        yt.append(y - 0.28 * 2); yl.append(lab); y += 0.7
    ax.axvline(0, color=INK2, lw=0.9); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.set_xlabel('BA difference (points, 95% CI); positive favours the sequential rule')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker=mk, color=col, lw=1.6, ms=5.5, label=nm) for nm, col, mk, _ in style], frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT / 'figures/fig3_policy_contrasts.png', dpi=200); plt.close(fig)


fig1(); fig2(); fig3()
(OUT / 'numbers.json').write_text(json.dumps(NUM, indent=1), encoding='utf-8')
print('tables:', sorted(p.name for p in (OUT / 'tables').glob('*.md'))); print('figures:', sorted(p.name for p in (OUT / 'figures').glob('*.png')))
