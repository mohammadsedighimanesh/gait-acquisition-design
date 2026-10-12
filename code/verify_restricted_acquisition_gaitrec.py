"""Independent verifier for gaitrec_pilot_v1 / gaitrec_grid_v1 (Addendum A3). No refits. Usage: python verify_restricted_acquisition_gaitrec.py pilot|grid"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
SRC = ROOT / 'idea1_2026/gaitrec_trial_stability_pilot_v1'
ALLOC = {300: (33, 54, 85, 81, 47), 500: (56, 90, 141, 135, 78), 600: (67, 108, 170, 161, 94), 800: (90, 144, 226, 215, 125), 1000: (112, 180, 283, 269, 156)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main(stage):
    out = STUDY / f'gaitrec_{stage}_v1'; ch = {}
    fr = json.loads((STUDY / 'frozen_protocol_A3.json').read_text())
    ch['addendum_hash'] = sha(STUDY / 'ADDENDUM_A3_GAITREC.md') == fr['addendum_A3_sha256']
    ch['script_hash'] = sha(ROOT / 'restricted_acquisition_gaitrec.py') == fr['script_sha256']
    ch['input_hashes'] = all(sha(ROOT / p) == h for p, h in fr['input_sha256'].items())
    man = json.loads((out / 'output_manifest.json').read_text()); ch['manifest_all_hashed'] = all(sha(out / n) == h for n, h in man.items())
    prog = json.loads((out / 'progress.json').read_text()); expected = 50 if stage == 'pilot' else 4500
    ch['fit_counts'] = prog['primary_fits'] == expected and prog['exclusion_control_fits'] == expected and sum(n.startswith('models/') for n in man) == expected
    bags = np.load(SRC / 'development_bags.npz'); X = bags['X']; y = bags['y'].astype(int); pids = bags['person_ids']
    meta = pd.read_csv(ROOT / 'idea1_2026/gaitrec_source_vertical_v1/GRF_metadata.csv'); test_ids = set(meta.loc[meta.TEST == 1, 'SUBJECT_ID'])
    mem = pd.read_csv(out / 'acquisition_memberships.csv'); pred = pd.read_csv(out / 'heldout_predictions.csv'); audit = pd.read_csv(out / 'acquisition_audit.csv')
    conv = pd.read_csv(out / 'convergence_audit.csv'); splits = pd.read_csv(SRC / 'outer_person_splits_before_fit.csv')
    ch['no_official_test_person'] = not (set(mem.person_id) | set(pred.person_id)) & test_ids
    ch['exclusion_flags_all_true'] = bool(audit.exclusion_proof_identical.all()); ch['class_order_all'] = bool((audit.class_order == '0,1,2,3,4').all())
    ch['convergence_no_max_iter'] = int(conv.reached_max_iter.sum()) == 0; ch['convergence_warnings'] = int(conv.convergence_warning.sum())
    train = mem[mem.role == 'train']; test = mem[mem.role == 'test']
    nested = True; classes_ok = True; disjoint = True; replay = 0.0; excl = 0.0; models = 0; five = True
    for (r, f), g in train.groupby(['repeat', 'fold']):
        te_rec = set(splits[(splits.repeat == r) & (splits.fold == f) & (splits.role == 'test')].person_id)
        tr_rec = set(splits[(splits.repeat == r) & (splits.fold == f) & (splits.role == 'train')].person_id)
        te_saved = test[(test.repeat == r) & (test.fold == f)]; disjoint &= set(te_saved.person_id) == te_rec
        te_idx = te_saved.sort_values('person_index').person_index.to_numpy(); tf = np.median(X[te_idx], axis=1)
        for d, gd in g.groupby('draw'):
            ns_here = sorted(gd.n.unique()); prev_people = None
            for n in ns_here:
                prev_trials = None
                for k in sorted(gd[gd.n == n].k.unique()):
                    cell = gd[(gd.n == n) & (gd.k == k)]; people = set(cell.person_id)
                    classes_ok &= tuple(np.bincount(cell['class'].to_numpy(), minlength=5)) == ALLOC[n]; five &= cell['class'].nunique() == 5
                    disjoint &= not people & te_rec and people <= tr_rec
                    trials = {int(p): set(map(int, t.split(','))) for p, t in zip(cell.person_index, cell.retained_trials)}
                    nested &= all(len(t) == k for t in trials.values())
                    if prev_trials is not None:
                        nested &= all(prev_trials[p] <= trials[p] for p in trials)
                    prev_trials = trials
                    m = np.load(out / 'models' / f'r{r}_f{f}_d{d}_n{n}_k{k}.npz'); models += 1
                    trf = np.stack([np.median(X[p, sorted(np.array(list(trials[p])) - 1)], axis=0) for p in sorted(trials)])
                    sd = trf.std(0); excl = max(excl, float(np.abs(trf.mean(0) - m['s1_mean']).max()), float(np.abs(np.where(sd > 0, sd, 1.0) - m['s1_scale']).max()))
                    z = (tf - m['s1_mean']) / m['s1_scale']; z = (z - m['pca_mean']) @ m['pca_components'].T; z = (z - m['s2_mean']) / m['s2_scale']
                    p_hat = softmax(z @ m['coef'].T + m['intercept'])
                    saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.draw == d) & (pred.n == n) & (pred.k == k)].sort_values('person_index')
                    replay = max(replay, float(np.abs(p_hat - saved[[f'p{c}' for c in range(5)]].to_numpy()).max()))
                if prev_people is not None:
                    nested &= prev_people <= set(gd[(gd.n == n)].person_id)
                prev_people = set(gd[gd.n == n].person_id)
            for k in sorted(gd.k.unique()):
                ref = None
                for n in ns_here:
                    cur = dict(zip(gd[(gd.n == n) & (gd.k == k)].person_id, gd[(gd.n == n) & (gd.k == k)].retained_trials))
                    if not cur:
                        continue  # cell absent in this stage (pilot runs only two cells)
                    if ref is not None:
                        nested &= set(ref) <= set(cur) and all(ref[p] == cur[p] for p in ref)
                    ref = cur
    ch.update({'nested_people_and_trials': bool(nested), 'class_counts_match_table': bool(classes_ok), 'five_classes_present': bool(five), 'test_recorded_and_disjoint': bool(disjoint),
               'models_replayed': models, 'max_probability_replay_error': replay, 'replay_within_1e-8': replay < 1e-8,
               'scaler1_from_retained_trials_max_error': excl, 'exclusion_within_1e-8': excl < 1e-8})
    import restricted_acquisition_gaitrec as g
    if stage == 'grid':
        surface, con, fin, _ = g.summarize(pred, y, [(n, k) for n in g.NS for k in g.KS], g.CONTRASTS, g.DRAWS)
    else:
        surface, con, fin, _ = g.summarize(pred, y, [(1000, 3), (600, 5)], [g.CONTRASTS[0]], 5)
    s0 = pd.read_csv(out / 'learning_surface.csv'); c0 = pd.read_csv(out / 'prespecified_contrasts.csv')
    ch['surface_recomputed_match'] = bool(np.allclose(surface[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(), s0[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(), atol=1e-12))
    ch['contrast_recomputed_match'] = bool(np.allclose(con[['mean', 'ci_low', 'ci_high']].to_numpy(), c0[['mean', 'ci_low', 'ci_high']].to_numpy(), atol=1e-12))
    rows = {}
    for (r, d, n, k), gg in pred.groupby(['repeat', 'draw', 'n', 'k']):
        hard = gg[[f'p{c}' for c in range(5)]].to_numpy().argmax(1); lab = gg.label.to_numpy(); rows[(r, d, n, k)] = np.mean([(hard[lab == c] == c).mean() for c in range(5)])
    reps = sorted(pred.repeat.unique()); dr = sorted(pred.draw.unique())
    plain = float(np.mean([rows[(r, d, 1000, 3)] - rows[(r, d, 600, 5)] for r in reps for d in dr]))
    ch['primary_point_plain_recompute_match'] = abs(plain - float(c0[(c0.contrast == 'primary_1000x3_minus_600x5') & (c0.metric == 'balanced_accuracy')]['mean'].iloc[0])) < 1e-12
    bools = {k: v for k, v in ch.items() if isinstance(v, bool)}
    rep = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'stage': stage, 'checks': ch, 'verification_model_fits': 0}
    (out / 'independent_verification.json').write_text(json.dumps(rep, indent=2, default=float)); print(json.dumps(rep, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
