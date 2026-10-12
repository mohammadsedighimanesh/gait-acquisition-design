"""Check that every headline number quoted in the manuscript text equals the value in the verified result files. Prints PASS/FAIL per item."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
D = ROOT / 'Paper/Manuscript_Acquisition_Design_2026_10_12'
S = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'; T3 = ROOT / 'idea1_2026/stepup_third_task_v1'
txt = ((D / 'manuscript.md').read_text(encoding='utf-8') + (D / 'supplementary_information.md').read_text(encoding='utf-8')).replace('−', '-')
N = json.loads((D / 'numbers.json').read_text(encoding='utf-8'))
results = []


def sg(x, d=1):
    v = round(100 * x, d)
    return f'{0.0:+.{d}f}' if v == 0 else f'{v:+.{d}f}'


def un(x, d=1):
    return f'{abs(round(100 * x, d)):.{d}f}' if abs(round(100 * x, d)) != 0 else f'{0.0:.{d}f}'


def sn(x, d=1):  # signed number without plus for negatives text e.g. -1.3
    return f'{round(100 * x, d):.{d}f}'


def chk(name, snippet):
    ok = snippet.replace('−', '-') in txt
    results.append((ok, name, snippet))


# headline contrasts (abstract and section 3.2)
for nm, key, label in [('FPI', 'FPI_primary', 'FPI'), ('GaitRec', 'GaitRec_primary_1000x3_minus_600x5', 'GaitRec'), ('BMI', 'StepUP_bmi_primary_120x5_minus_60x10', 'BMI'), ('sex', 'StepUP_sex_primary_120x5_minus_60x10', 'sex')]:
    v = N[key]; m, lo, hi = v['mean'], v['lo'], v['hi']
    chk(f'abstract {nm}', (f'{sg(m)} points, 95% CI {sn(lo)} to {sn(hi)}' if nm == 'FPI' else f'{sg(m)}, {sn(lo)} to {sn(hi)}'))
chk('sec3.2 FPI', f'by {un(N["FPI_primary"]["mean"])} points (95% CI {sn(N["FPI_primary"]["lo"])} to {sn(N["FPI_primary"]["hi"])}')
g = N['GaitRec_primary_1000x3_minus_600x5']; chk('sec3.2 GaitRec primary', f'was {un(g["mean"])} points ({sn(g["lo"])} to {sn(g["hi"])})')
g = N['GaitRec_secondary_500x3_minus_300x5']; chk('sec3.2 GaitRec secondary', f'it was {un(g["mean"])} ({sn(g["lo"])} to {sn(g["hi"])})')
g = N['StepUP_bmi_primary_120x5_minus_60x10']; chk('sec3.2 BMI', f'by {un(g["mean"])} points ({sn(g["lo"])} to {sn(g["hi"])})')
g = N['StepUP_sex_primary_120x5_minus_60x10']; chk('sec3.2 sex', f'({sg(g["mean"])}, {sn(g["lo"])} to {sn(g["hi"])};')
f = N['FPI_secondary']; chk('sec3.2 FPI near-budget', f'gave {sg(f["mean"])} points ({sn(f["lo"])} to {sn(f["hi"])})')
for nm, key, ll in [('FPI', 'FPI_primary_logloss', 'mean'), ('GaitRec', 'GaitRec_primary_1000x3_minus_600x5', 'll_mean'), ('sex', 'StepUP_sex_primary_120x5_minus_60x10', 'll_mean'), ('BMI', 'StepUP_bmi_primary_120x5_minus_60x10', 'll_mean')]:
    v = N[key][ll]; chk(f'logloss {nm}', f'{nm if nm in ("FPI",) else ("GaitRec" if nm == "GaitRec" else ("StepUP sex" if nm == "sex" else "StepUP BMI"))} {v:.3f}')
# sec 3.3
a = N['StepUP_sex_120_5_10']; chk('3.3 sex 5-10', f'added {un(a["mean"])} points ({sn(a["lo"])} to {sn(a["hi"])})')
a = N['StepUP_sex_120_10_20']; chk('3.3 sex 10-20', f'added {un(a["mean"])} points ({sn(a["lo"])} to {sn(a["hi"])})')
a = N['StepUP_bmi_120_10_20']; chk('3.3 BMI 10-20', f'{sn(a["mean"])} points, 95% CI {sn(a["lo"])} to {sn(a["hi"])}')
# variability
chk('3.4 within SD', f'{100*N["FPI_primary_var"]["within"]:.1f} points'); chk('3.4 between SD', f'{100*N["FPI_primary_var"]["between"]:.2f} points')
chk('3.4 21x3 SD', f'{100*N["FPI_var_21x3_within"]:.1f} points at 21 × 3')
# sensitivity
a = N['sens_FPI_tuned']; chk('3.5 tuned', f'{sg(a["mean"])} points ({sn(a["lo"])} to {sn(a["hi"])}) with tuned C')
chk('3.5 C share 35', f'{round(100*N["sens_FPI_C01_share"]["35"])}% of the 35 × 3'); chk('3.5 C share 21', f'{round(100*N["sens_FPI_C01_share"]["21"])}% of the 21 × 5')
for cell, v in N['sens_FPI_ll_worse'].items():
    chk(f'3.5 ll {cell}', f'{v[0]:.3f} ({cell.replace("x", " × ")})')
# inference-side
a = N['inference_median_5_1']; chk('S10 5 vs 1', f'{un(a[0])} points (95% CI {sn(a[1])} to {sn(a[2])})')
a = N['inference_median_3_5']; chk('S10 3 vs 5', f'{un(a[0])} points below five ({sn(a[1])} to {sn(a[2])})')
# policy
P = {k: v for k, v in N.items() if k.startswith('policy_') and isinstance(v, dict) and 'conv5' in v}
def pol(key):
    return P[key]
for key, label in [('policy_FPI_A4_105', 'FPI105'), ('policy_FPI_A4_70', 'FPI70'), ('policy_GaitRec_A4_1500', 'Gait1500'), ('policy_StepUP BMI_A5_300', 'BMI300'), ('policy_StepUP BMI_A5.2_150', 'BMI150'), ('policy_StepUP BMI_A5.2_100', 'BMI100'),
                   ('policy_StepUP sex_A5_300', 'sex300'), ('policy_StepUP sex_A5.2_150', 'sex150'), ('policy_StepUP sex_A5.2_100', 'sex100')]:
    v = pol(key)['conv5']; chk(f'policy conv5 {label}', f'{sg(v[0])}' if 'sex' in label else (f'{sg(v[0])} points (95% CI {sn(v[1])} to {sn(v[2])})' if label == 'FPI105' else f'{sg(v[0])} ({sn(v[1])} to {sn(v[2])})'))
v = pol('policy_GaitRec_A4_3000')['conv5']; chk('policy gaitrec 3000', f'gained {un(v[0])} point ({sn(v[1])} to {sn(v[2])})')
for key, label, who in [('policy_FPI_A4_70', 'FPI70', 'FPI at 70 recordings'), ('policy_GaitRec_A4_1500', 'Gait1500', 'GaitRec at 1,500'), ('policy_StepUP BMI_A5.2_100', 'BMI100', 'StepUP BMI at 100')]:
    v = pol(key)['best']; chk(f'best-fixed {label}', f'({sg(v[0])}, {sn(v[1])} to {sn(v[2])})' if label != 'FPI70' else f'{sg(v[0])} points, {sn(v[1])} to {sn(v[2])})')
v = pol('policy_FPI_A4_105')['random']; chk('random FPI105', f'{sg(v[0])}, {sn(v[1])} to {sn(v[2])})')
v = pol('policy_StepUP BMI_A5.2_100')['random']; chk('random BMI100', f'{sg(v[0])}, {sn(v[1])} to {sn(v[2])})')
v = pol('policy_StepUP sex_A5.2_150')['best']; chk('sex150 best', f'{un(v[0])} points below the best fixed design ({sn(v[1])} to {sn(v[2])})')
v = pol('policy_StepUP sex_A5.2_150')['conv5']; chk('sex150 conv', f'{un(v[0])} points below the convention ({sn(v[1])} to {sn(v[2])})')
rng = [min(P[k]['best'][0] for k in P if not P[k]['saturated']), max(P[k]['best'][0] for k in P)]
chk('best fixed range', f'ranged from {sn(rng[0])} to {sg(rng[1])} points')
chk('share people', f'{round(100*N["policy_share_people_range"][0])}% to {round(100*N["policy_share_people_range"][1])}%')
# totals
t = N['totals']; chk('final fits', f'{t["primary"] + t["final"]:,} final model fits'); chk('internal', f'{t["internal"]:,} policy-internal'); chk('inner', f'{t["inner"]:,} inner-tuning'); chk('refits', f'{t["refit"]:,} refits'); chk('controls', f'{t["control"]:,} exclusion-control fits')
# finite-difference counts and 3->5 counts (recomputed)
tot = {'n': [0, 0, 0, 0], 'k': [0, 0, 0, 0]}
for d in [S / 'grid_v1', S / 'gaitrec_grid_v1', T3 / 'surface_L1_v1', T3 / 'surface_L2_v1']:
    f_ = pd.read_csv(d / 'finite_differences.csv')
    for ax in ['n', 'k']:
        g_ = f_[f_.axis == ax]; tot[ax][0] += len(g_); tot[ax][1] += int((g_.mean_difference > 0).sum()); tot[ax][2] += int((g_.ci_low > 0).sum()); tot[ax][3] += int((g_.ci_high < 0).sum())
chk('fd people', f'{tot["n"][1]} of {tot["n"][0]} steps that added people had a positive point estimate and {tot["n"][2]} had a 95% interval above zero'); chk('fd trials', f'{tot["k"][1]} of {tot["k"][0]} were positive, {tot["k"][2]} had an interval above zero')
assert tot['n'][3] == 0 and tot['k'][3] == 0
cells = []
for d in [S / 'grid_v1', S / 'gaitrec_grid_v1', T3 / 'surface_L1_v1', T3 / 'surface_L2_v1']:
    f_ = pd.read_csv(d / 'finite_differences.csv'); g_ = f_[(f_.axis == 'k') & f_['from'].str.endswith('x3') & f_['to'].str.endswith('x5')]; cells += list(zip(g_.mean_difference, g_.ci_low, g_.ci_high))
inside = [(m, lo, hi) for m, lo, hi in cells if -0.0015 <= m <= 0.0105]
chk('3to5 counts', f'In {len(inside)} of the {len(cells)} cells'); chk('3to5 zero', f'in {sum(1 for m, lo, hi in cells if lo <= 0 <= hi)} the interval included zero')
# surface ranges
for nm, d, lo_key, hi_key in [('FPI', S / 'grid_v1', '21x1', '56x5'), ('GaitRec', S / 'gaitrec_grid_v1', '300x1', '1000x5'), ('sex', T3 / 'surface_L1_v1', '30x1', '120x20'), ('bmi', T3 / 'surface_L2_v1', '30x1', '120x5')]:
    s_ = pd.read_csv(d / 'learning_surface.csv'); a = s_[(s_.n == int(lo_key.split('x')[0])) & (s_.k == int(lo_key.split('x')[1]))].balanced_accuracy.iloc[0]; b = s_[(s_.n == int(hi_key.split('x')[0])) & (s_.k == int(hi_key.split('x')[1]))].balanced_accuracy.iloc[0]
    chk(f'surface {nm}', f'{a:.3f}'); chk(f'surface {nm} hi', f'{b:.3f}')
bad = [r for r in results if not r[0]]
for ok, name, snip in results:
    print('PASS' if ok else 'FAIL', '|', name, '|', snip)
print(f'\n{len(results) - len(bad)}/{len(results)} checks pass')
(D / 'number_audit.txt').write_text('\n'.join(('PASS' if ok else 'FAIL') + ' | ' + name + ' | ' + snip for ok, name, snip in results) + f'\n\n{len(results) - len(bad)}/{len(results)} checks pass\n', encoding='utf-8')
