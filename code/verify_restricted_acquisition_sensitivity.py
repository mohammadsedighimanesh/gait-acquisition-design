"""Independent verifier for sensitivity_v1 (Addendum A2). No refits. PASS covers only listed checks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
GRID = STUDY / 'grid_v1'; OUT = STUDY / 'sensitivity_v1'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
CS = [0.1, 1.0, 10.0]; CELLS = [(35, 3), (21, 5)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ch = {}
    fr = json.loads((STUDY / 'frozen_protocol_A2.json').read_text())
    ch['addendum_hash'] = sha(STUDY / 'ADDENDUM_A2_SENSITIVITY_TUNING.md') == fr['addendum_A2_sha256']
    ch['script_hash'] = sha(ROOT / 'restricted_acquisition_sensitivity.py') == fr['script_sha256']
    ch['input_hashes'] = all(sha(ROOT / p) == h for p, h in fr['input_sha256'].items())
    ch['grid_inputs_unchanged'] = sha(GRID / 'acquisition_memberships.csv') == fr['grid_memberships_sha256'] and sha(GRID / 'heldout_predictions.csv') == fr['grid_predictions_sha256']
    man = json.loads((OUT / 'output_manifest.json').read_text())
    ch['manifest_all_hashed'] = all(sha(OUT / n) == h for n, h in man.items())
    ch['manifest_models_600'] = sum(n.startswith('models/') for n in man) == 600
    prog = json.loads((OUT / 'progress.json').read_text())
    inner = pd.read_csv(OUT / 'inner_tuning_losses.csv'); sel = pd.read_csv(OUT / 'selected_C.csv'); imem = pd.read_csv(OUT / 'inner_fold_memberships.csv')
    pred = pd.read_csv(OUT / 'heldout_predictions_tuned.csv'); conv = pd.read_csv(OUT / 'convergence_audit.csv')
    ch['fit_counts'] = prog['inner_fits'] == 5400 and prog['refits'] == 600 and prog['exclusion_control_fits'] == 600 and len(inner) == 5400 and len(sel) == 600
    ch['convergence_no_max_iter'] = int(conv.reached_max_iter.sum()) == 0; ch['convergence_warnings'] = int(conv.convergence_warning.sum())
    # Selection rule replay.
    ok = True
    for _, r in sel.iterrows():
        # r['repeat'] not r.repeat: Series.repeat is a method.
        g = inner[(inner.repeat == r['repeat']) & (inner.fold == r['fold']) & (inner.draw == r['draw']) & (inner.n == r['n']) & (inner.k == r['k'])]
        means = g.groupby('C').inner_val_log_loss.mean(); assert len(g) == 9
        best = min(CS, key=lambda C: (means[C], C)); ok &= best == r['selected_C']
        ok &= all(abs(means[C] - r[f'mean_inner_ll_C{C}']) < 1e-12 for C in CS)
    ch['selection_rule_replayed'] = bool(ok)
    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    mem = pd.read_csv(GRID / 'acquisition_memberships.csv'); test_mem = mem[mem.role == 'test']
    part_ok = True; replay = 0.0; excl = 0.0; models = 0; mem_same = True
    for (r, f, d, n, k), g in imem.groupby(['repeat', 'fold', 'draw', 'n', 'k']):
        cell = mem[(mem.role == 'train') & (mem.repeat == r) & (mem.fold == f) & (mem.draw == d) & (mem.n == n) & (mem.k == k)]
        acquired = set(cell.participant_id); te = set(test_mem[(test_mem.repeat == r) & (test_mem.fold == f)].participant_id)
        for i, gi in g.groupby('inner_fold'):
            tr = set(gi[gi.role == 'inner_train'].participant_id); va = set(gi[gi.role == 'inner_val'].participant_id)
            part_ok &= (tr | va) == acquired and not tr & va and not (tr | va) & te
        vals = g[g.role == 'inner_val'].groupby('participant_id').size(); part_ok &= set(vals.index) == acquired and (vals == 1).all()
        m = np.load(OUT / 'models' / f'r{r}_f{f}_d{d}_n{n}_k{k}.npz'); models += 1
        trials = {int(p): [int(t) - 1 for t in ts.split(',')] for p, ts in zip(cell.participant_id, cell.retained_trials)}
        trf = np.stack([np.concatenate([np.median(X[p - 1, trials[p]], axis=0), A[p - 1]]) for p in sorted(trials)])
        sd = trf.std(0); excl = max(excl, float(np.abs(trf.mean(0) - m['scale_mean']).max()), float(np.abs(np.where(sd > 0, sd, 1.0) - m['scale_scale']).max()))
        tei = np.array(sorted(te)) - 1; tf = np.concatenate([np.median(X[tei], axis=1), A[tei]], axis=1)
        p = softmax(((tf - m['scale_mean']) / m['scale_scale']) @ m['coef'].T + m['intercept'])
        saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.draw == d) & (pred.n == n) & (pred.k == k)].sort_values('participant_id')
        replay = max(replay, float(np.abs(p - saved[['p0', 'p1', 'p2']].to_numpy()).max()))
        mem_same &= float(m['C'][0]) == float(saved.selected_C.iloc[0]) == float(sel[(sel.repeat == r) & (sel.fold == f) & (sel.draw == d) & (sel.n == n) & (sel.k == k)].selected_C.iloc[0])
    ch.update({'inner_folds_partition_acquired_people_disjoint_from_test': bool(part_ok), 'models_replayed': models, 'max_probability_replay_error': replay,
               'replay_within_1e-8': replay < 1e-8, 'scaler_from_retained_trials_max_error': excl, 'exclusion_within_1e-8': excl < 1e-8,
               'selected_C_consistent_across_files': bool(mem_same)})
    import restricted_acquisition_sensitivity as s
    grid_pred = pd.read_csv(GRID / 'heldout_predictions.csv')
    rec = s.summarize(pred, grid_pred, sel, y)
    for name in ['tuned_contrasts', 'tuned_vs_fixed_by_cell']:
        a = rec[name].select_dtypes('number').to_numpy(); b = pd.read_csv(OUT / f'{name}.csv').select_dtypes('number').to_numpy()
        ch[f'{name}_recomputed_match'] = bool(np.allclose(a, b, atol=1e-12))
    bools = {k: v for k, v in ch.items() if isinstance(v, bool)}
    rep = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'checks': ch, 'verification_model_fits': 0}
    (OUT / 'independent_verification.json').write_text(json.dumps(rep, indent=2, default=float)); print(json.dumps(rep, indent=2, default=float))


if __name__ == '__main__':
    main()
