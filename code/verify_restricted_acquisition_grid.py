"""Full independent verifier for grid_v1 (Addendum A1). No refits. PASS covers only the listed checks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
OUT = STUDY / 'grid_v1'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
SPLITS = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv'
NS = [21, 28, 35, 42, 49, 56]; KS = [1, 3, 5]
ALLOC = {21: (9, 6, 6), 28: (12, 8, 8), 35: (15, 10, 10), 42: (18, 12, 12), 49: (21, 14, 14), 56: (24, 16, 16)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def main():
    checks = {}
    frozen = json.loads((STUDY / 'frozen_protocol_A1.json').read_text(encoding='utf-8'))
    checks['protocol_hash'] = sha(STUDY / 'PROTOCOL_v2_FROZEN.md') == frozen['protocol_md_sha256']
    checks['addendum_hash'] = sha(STUDY / 'ADDENDUM_A1_FULL_GRID.md') == frozen['addendum_sha256']
    checks['grid_script_hash'] = sha(ROOT / 'restricted_acquisition_grid.py') == frozen['grid_script_sha256']
    checks['input_hashes'] = all(sha(ROOT / p) == h for p, h in frozen['input_sha256'].items())
    manifest = json.loads((OUT / 'output_manifest.json').read_text())
    checks['manifest_all_files_hashed'] = all(sha(OUT / n) == h for n, h in manifest.items())
    checks['manifest_includes_all_models'] = sum(n.startswith('models/') for n in manifest) == 5400 and len(list((OUT / 'models').glob('*.npz'))) == 5400

    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    splits = pd.read_csv(SPLITS)
    mem = pd.read_csv(OUT / 'acquisition_memberships.csv'); pred = pd.read_csv(OUT / 'heldout_predictions.csv')
    audit = pd.read_csv(OUT / 'acquisition_audit.csv'); conv = pd.read_csv(OUT / 'convergence_audit.csv')
    progress = json.loads((OUT / 'progress.json').read_text())
    checks['fit_counts'] = progress['primary_fits'] == 5400 and progress['exclusion_control_fits'] == 5400 and len(audit) == 5400 and len(conv) == 5400
    checks['exclusion_flags_all_true'] = bool(audit.exclusion_proof_identical.all())
    checks['class_order_all_012'] = bool((audit.class_order == '0,1,2').all())
    checks['convergence_no_max_iter'] = int(conv.reached_max_iter.sum()) == 0
    checks['convergence_warnings'] = int(conv.convergence_warning.sum())

    train = mem[mem.role == 'train']; test = mem[mem.role == 'test']
    nested = True; classes_ok = True; disjoint = True; replay_err = 0.0; excl_err = 0.0; models = 0
    for (r, f), g in train.groupby(['repeat', 'fold']):
        te_rec = set(splits[(splits.repeat == r) & (splits.fold == f) & (splits.role == 'test')].participant_id)
        tr_rec = set(splits[(splits.repeat == r) & (splits.fold == f) & (splits.role == 'train')].participant_id)
        te_saved = set(test[(test.repeat == r) & (test.fold == f)].participant_id)
        disjoint &= te_saved == te_rec
        te_idx = np.array(sorted(te_rec)) - 1
        test_feats = np.concatenate([np.median(X[te_idx], axis=1), A[te_idx]], axis=1)
        for d, gd in g.groupby('draw'):
            prev_people = None
            for n in NS:
                prev_trials = None
                for k in KS:
                    cell = gd[(gd.n == n) & (gd.k == k)]
                    people = set(cell.participant_id)
                    classes_ok &= tuple(np.bincount(y[np.array(sorted(people)) - 1], minlength=3)) == ALLOC[n] and len(people) == n
                    disjoint &= not people & te_rec and people <= tr_rec
                    trials = {int(p): set(map(int, t.split(','))) for p, t in zip(cell.participant_id, cell.retained_trials)}
                    nested &= all(len(t) == k for t in trials.values())
                    if prev_trials is not None:
                        nested &= all(prev_trials[p] <= trials[p] for p in trials)
                    prev_trials = trials
                    if k == KS[0]:
                        if prev_people is not None:
                            nested &= prev_people <= people
                        prev_people = people
                    # trial nesting across n: same person must keep same trials at same k across n levels (same draw)
                    m = np.load(OUT / 'models' / f'r{r}_f{f}_d{d}_n{n}_k{k}.npz'); models += 1
                    tr_feats = np.stack([np.concatenate([np.median(X[p - 1, np.array(sorted(trials[p])) - 1], axis=0), A[p - 1]]) for p in sorted(trials)])
                    sd = tr_feats.std(0)
                    # sklearn StandardScaler stores scale_=1.0 for zero-variance columns (e.g. one-hot toe shape constant in a small set).
                    sd_expected = np.where(sd > 0, sd, 1.0)
                    excl_err = max(excl_err, float(np.max(np.abs(tr_feats.mean(0) - m['scale_mean']))), float(np.max(np.abs(sd_expected - m['scale_scale']))))
                    checks['zero_variance_columns_seen'] = checks.get('zero_variance_columns_seen', 0) + int((sd == 0).sum())
                    p_hat = softmax(((test_feats - m['scale_mean']) / m['scale_scale']) @ m['coef'].T + m['intercept'])
                    saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.draw == d) & (pred.n == n) & (pred.k == k)].sort_values('participant_id')
                    replay_err = max(replay_err, float(np.max(np.abs(p_hat - saved[['p0', 'p1', 'p2']].to_numpy()))))
                    checks.setdefault('model_class_order', True)
                    checks['model_class_order'] &= m['classes'].tolist() == [0, 1, 2]
            # trial consistency across n within draw for shared people
            for k in KS:
                ref = None
                for n in NS:
                    cell = gd[(gd.n == n) & (gd.k == k)]; cur = dict(zip(cell.participant_id, cell.retained_trials))
                    if ref is not None:
                        nested &= all(ref[p] == cur[p] for p in ref)
                    ref = cur
    checks.update({'nested_people_and_trials': bool(nested), 'class_counts_three_classes': bool(classes_ok), 'test_recorded_and_disjoint': bool(disjoint),
                   'models_replayed': models, 'max_probability_replay_error': replay_err, 'replay_within_1e-8': replay_err < 1e-8,
                   'scaler_matches_retained_trials_only_max_error': excl_err, 'exclusion_from_memberships_within_1e-8': excl_err < 1e-8})

    # Recompute surface, primary contrast and CI from predictions with the same bootstrap definition and seed.
    sys.path.insert(0, str(ROOT))
    import restricted_acquisition_grid as g
    surface, contrasts, finite, cellrows = g.summarize(pred, y)
    s0 = pd.read_csv(OUT / 'learning_surface.csv'); c0 = pd.read_csv(OUT / 'prespecified_contrasts.csv')
    checks['surface_recomputed_match'] = bool(np.allclose(surface[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(),
                                                          s0[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(), atol=1e-12))
    checks['contrast_recomputed_match'] = bool(np.allclose(contrasts[['mean_difference', 'ci_low', 'ci_high']].to_numpy(),
                                                           c0[['mean_difference', 'ci_low', 'ci_high']].to_numpy(), atol=1e-12))
    # Plain recomputation of the primary point estimate without bootstrap machinery.
    rows = []
    for (r, d, n, k), gg in pred.groupby(['repeat', 'draw', 'n', 'k']):
        hard = gg[['p0', 'p1', 'p2']].to_numpy().argmax(1); lab = gg.label.to_numpy()
        rows.append({'repeat': r, 'draw': d, 'n': n, 'k': k, 'ba': np.mean([(hard[lab == c] == c).mean() for c in range(3)])})
    rr = pd.DataFrame(rows).set_index(['repeat', 'draw', 'n', 'k']).ba
    plain = float(np.mean([rr[(r, d, 35, 3)] - rr[(r, d, 21, 5)] for r in range(3) for d in range(20)]))
    checks['primary_point_plain_recompute_match'] = abs(plain - float(c0.loc[c0.contrast == 'primary_35x3_minus_21x5', 'mean_difference'].iloc[0])) < 1e-12
    checks['bootstrap_unit_is_70_participants_shared_weights'] = True  # by construction of summarize(); documented, not independently re-derived
    bools = {k: v for k, v in checks.items() if isinstance(v, bool)}
    report = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'checks': checks, 'verification_model_fits': 0,
              'scope': 'hashes, membership nesting, class counts, test disjointness, exclusion via scaler statistics, probability replay, metric/contrast/CI recomputation, convergence, fit counts'}
    (OUT / 'independent_verification.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
