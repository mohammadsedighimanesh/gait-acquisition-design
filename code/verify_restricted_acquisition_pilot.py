"""Independent verifier for the restricted-acquisition pilot. Replays saved models without refitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
OUT = STUDY / 'pilot_v1'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
SPLITS = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv'
ALLOC = {21: (9, 6, 6), 35: (15, 10, 10)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def main():
    report = {'status': 'FAIL', 'checks': {}}
    frozen = json.loads((STUDY / 'frozen_protocol.json').read_text(encoding='utf-8'))
    report['checks']['protocol_md_hash'] = sha(STUDY / 'PROTOCOL_v2_FROZEN.md') == frozen['protocol_md_sha256']
    report['checks']['input_hashes'] = all(sha(ROOT / p) == h for p, h in frozen['input_sha256'].items())
    manifest = json.loads((OUT / 'output_manifest.json').read_text())
    report['checks']['output_manifest'] = all(sha(OUT / n) == h for n, h in manifest.items() if n != 'output_manifest.json')

    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    splits = pd.read_csv(SPLITS); splits = splits[splits.repeat == 0]
    mem = pd.read_csv(OUT / 'acquisition_memberships.csv'); pred = pd.read_csv(OUT / 'heldout_predictions.csv')
    audit = pd.read_csv(OUT / 'acquisition_audit.csv')

    nested_ok = True; class_ok = True; disjoint_ok = True; replay_err = 0.0; models = 0; three_classes = True
    for (fold, draw), g in mem.groupby(['fold', 'draw']):
        te_rec = set(splits[(splits.fold == fold) & (splits.role == 'test')].participant_id)
        tr_rec = set(splits[(splits.fold == fold) & (splits.role == 'train')].participant_id)
        a = g[(g.design == 'A_35x3') & (g.role == 'train')]; b = g[(g.design == 'B_21x5') & (g.role == 'train')]
        nested_ok &= set(b.participant_id) <= set(a.participant_id)
        for _, r in b.iterrows():
            ta = set(a[a.participant_id == r.participant_id].retained_trials.iloc[0].split(','))
            nested_ok &= ta <= set(r.retained_trials.split(',')) and len(ta) == 3 and len(r.retained_trials.split(',')) == 5
        for d, n in [('A_35x3', 35), ('B_21x5', 21)]:
            tr = g[(g.design == d) & (g.role == 'train')]
            counts = tuple(np.bincount(y[tr.participant_id.to_numpy() - 1], minlength=3))
            class_ok &= counts == ALLOC[n]; three_classes &= len(set(y[tr.participant_id.to_numpy() - 1])) == 3
            te = set(g[(g.design == d) & (g.role == 'test')].participant_id)
            disjoint_ok &= te == te_rec and not te & set(tr.participant_id) and set(tr.participant_id) <= tr_rec
            # Replay: rebuild features from saved memberships only, apply saved scaler/coefs.
            m = np.load(OUT / 'models' / f'f{fold}_d{draw}_{d}.npz'); models += 1
            people = np.array(sorted(te)) - 1
            feats = np.concatenate([np.median(X[people], axis=1), A[people]], axis=1)
            z = ((feats - m['scale_mean']) / m['scale_scale']) @ m['coef'].T + m['intercept']
            p = softmax(z)
            saved = pred[(pred.fold == fold) & (pred.draw == draw) & (pred.design == d)].sort_values('participant_id')[['p0', 'p1', 'p2']].to_numpy()
            replay_err = max(replay_err, float(np.max(np.abs(p - saved))))
            # Training-feature replay: the scaler mean must equal the mean of the retained-trial medians.
            tr_people = tr.participant_id.to_numpy() - 1
            tr_feats = []
            for pid, ts in zip(tr_people, tr.retained_trials):
                idx = np.array([int(t) - 1 for t in ts.split(',')])
                tr_feats.append(np.concatenate([np.median(X[pid, idx], axis=0), A[pid]]))
            replay_err = max(replay_err, float(np.max(np.abs(np.mean(tr_feats, axis=0) - m['scale_mean']))))
    report['checks'].update({'nested_membership': bool(nested_ok), 'class_counts': bool(class_ok), 'all_three_classes': bool(three_classes),
                             'test_disjoint_and_recorded': bool(disjoint_ok), 'models_replayed': models,
                             'max_replay_error': replay_err, 'replay_within_1e-8': replay_err < 1e-8,
                             'exclusion_proof_all_true': bool(audit.exclusion_proof_identical.all()),
                             'provenance_replay_all_pass': bool(pd.read_csv(OUT / 'provenance_replay.csv')['pass'].all())})
    progress = json.loads((OUT / 'progress.json').read_text())
    report['checks']['fit_count_matches_authorization'] = progress['pilot_logistic_fits'] == 50 and progress['full_grid_run'] is False
    report['verification_model_fits'] = 0
    report['status'] = 'PASS_PILOT_MECHANICAL_CHECKS' if all(v is True for k, v in report['checks'].items() if isinstance(v, bool)) else 'FAIL'
    (OUT / 'independent_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
