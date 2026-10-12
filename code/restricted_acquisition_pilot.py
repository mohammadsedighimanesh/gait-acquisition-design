"""Mechanical feasibility pilot for the restricted-acquisition learning surface (repeat 0, cells 35x3 and 21x5, 5 draws, fixed C=0.1).

Authorized scope only. The full grid is not run here. No environment change, no original file modified.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
OUT = STUDY / 'pilot_v1'
PROTOCOL_MD = STUDY / 'PROTOCOL_v2_DRAFT.md'
PROTOCOL_JSON_DRAFT = STUDY / 'frozen_protocol_DRAFT.json'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
SPLITS = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv'
RAW_CSV = ROOT / 'idea1_2026/corrected_raw_pipeline_diagnostic_v2/raw_trial_level_features_v2.csv'
DATA_MAT = ROOT / 'dataset/DATA.mat'
EXPECTED = {CACHE: '1cdef1a7c3dc181f8e8ccfcdd1585992017d163ff79f6afa785b0e3cae37c924',
            SPLITS: '42b7d70a85da97ca6240b3fa4b85451cec5b13dae97ea1e6588eb463ffdd025c',
            RAW_CSV: '60e72d4a96d6600d23c4e0eda7d6255937e0da2ad11b7d3325ebcae1d47d30ed',
            DATA_MAT: '6fa44121f01ca0ca2843741a6169bb338fa554fa7835b18ebfcd6896c49e3374'}
SEED = 20261101
C_FIXED = 0.1
REPEAT = 0
DRAWS = 5
DESIGNS = {'A_35x3': (35, 3), 'B_21x5': (21, 5)}
ALLOC = {21: (9, 6, 6), 28: (12, 8, 8), 35: (15, 10, 10), 42: (18, 12, 12), 49: (21, 14, 14), 56: (24, 16, 16)}
FITS = {'pilot_logistic_fits': 0, 'exclusion_proof_duplicate_fits': 0}
ANGLE_NAMES = [f'R_{j}_{p}' for j in ['ANK', 'MT', 'MP'] for p in ['AB', 'INV', 'DF']]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def person_features(X_acq, A, people, trials):
    """Median over the explicitly retained trial indices only; abort if any NaN reaches the features."""
    med = np.stack([np.median(X_acq[p, trials[int(p)]], axis=0) for p in people])
    feats = np.concatenate([med, A[people]], axis=1)
    assert np.isfinite(feats).all(), 'NaN reached features'
    return feats


def fit_design(X_acq, A, y, people, trials, seed):
    feats = person_features(X_acq, A, people, trials)
    scaler = StandardScaler().fit(feats)
    model = LogisticRegression(C=C_FIXED, class_weight='balanced', max_iter=3000, tol=1e-7, random_state=seed)
    model.fit(scaler.transform(feats), y[people])
    return scaler, model


def predict(scaler, model, X, A, people):
    feats = np.concatenate([np.median(X[people], axis=1), A[people]], axis=1)
    return model.predict_proba(scaler.transform(feats))


def recompute_from_mat(pid, tid):
    people = np.atleast_1d(loadmat(DATA_MAT, squeeze_me=True, struct_as_record=False)['PARTICIPANT'])
    trial = np.atleast_1d(people[pid - 1].TRIAL)[tid - 1]
    row = []
    for name in ANGLE_NAMES:
        a = np.asarray(getattr(trial, name), dtype=float).reshape(-1)
        row.extend([a.mean(), a.max() - a.min()])
    pressure = np.asarray(trial.PRESS, dtype=float); positive = pressure[pressure > 0]; total = pressure.sum(axis=(0, 1))
    row.extend([float(np.asarray(trial.PMAX, dtype=float).mean()), positive.mean(), positive.std(), positive.max(), (pressure > 0).mean(),
                total.mean(), total.std(), total.max(), float(np.argmax(total) + 1), float(np.sum(total * np.arange(1, 101)) / total.sum())])
    return np.array(row)


def run():
    if OUT.exists():
        raise FileExistsError(f'Existing outputs preserved: {OUT}')
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, f'hash mismatch {path}'
    OUT.mkdir(parents=True); (OUT / 'models').mkdir()
    # Freeze protocol: copy md + json with hashes.
    protocol = json.loads(PROTOCOL_JSON_DRAFT.read_text(encoding='utf-8'))
    protocol['status'] = 'FROZEN_FOR_PILOT'
    protocol['protocol_md_sha256'] = sha(PROTOCOL_MD)
    protocol['pilot_script_sha256'] = sha(Path(__file__))
    protocol['input_sha256'] = {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()}
    protocol['software'] = {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}
    (STUDY / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    shutil.copy(PROTOCOL_MD, STUDY / 'PROTOCOL_v2_FROZEN.md')
    dump('frozen_protocol_copy.json', protocol)

    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    assert X.shape == (70, 5, 28) and A.shape == (70, 19) and np.bincount(y).tolist() == [30, 20, 20]
    splits = pd.read_csv(SPLITS); splits = splits[splits.repeat == REPEAT]

    # Check 6: provenance replay for 20 random trials against DATA.mat.
    rng0 = np.random.default_rng(SEED)
    prov = []
    for pid, tid in zip(rng0.integers(1, 71, 20), rng0.integers(1, 6, 20)):
        rec = recompute_from_mat(int(pid), int(tid)); diff = float(np.max(np.abs(rec - X[pid - 1, tid - 1])))
        prov.append({'participant_id': int(pid), 'trial_id': int(tid), 'max_abs_diff': diff, 'pass': diff <= 1e-9})
    pd.DataFrame(prov).to_csv(OUT / 'provenance_replay.csv', index=False)
    assert all(r['pass'] for r in prov), 'cache does not match DATA.mat recomputation'

    memberships = []; predictions = []; audits = []; timings = []
    for fold, block in splits.groupby('fold'):
        tr = np.sort(block.loc[block.role == 'train', 'participant_id'].to_numpy()) - 1
        te = np.sort(block.loc[block.role == 'test', 'participant_id'].to_numpy()) - 1
        assert len(tr) == 56 and len(te) == 14 and not set(tr) & set(te)
        assert tuple(np.bincount(y[tr], minlength=3)) == ALLOC[56]
        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + fold * 100 + draw)
            # Nested people: 35 (15/10/10) then 21 (9/6/6) subset of the 35.
            people35 = np.concatenate([rng.choice(tr[y[tr] == c], ALLOC[35][c], replace=False) for c in range(3)])
            people21 = np.concatenate([rng.choice(people35[y[people35] == c], ALLOC[21][c], replace=False) for c in range(3)])
            # Nested trials: 3 of 5 for design A; design B uses all 5 (superset) for its people.
            trials3 = {int(p): np.sort(rng.choice(5, 3, replace=False)) for p in people35}
            acquired = {'A_35x3': (np.sort(people35), {int(p): trials3[int(p)] for p in people35}),
                        'B_21x5': (np.sort(people21), {int(p): np.arange(5) for p in people21})}
            for design, (people, trials) in acquired.items():
                n, k = DESIGNS[design]
                assert len(people) == n and all(len(v) == k for v in trials.values())
                assert tuple(np.bincount(y[people], minlength=3)) == ALLOC[n] and len(set(y[people])) == 3
                assert not set(people) & set(te)
                # Build acquired array: NaN everywhere except retained trials.
                X_nan = np.full_like(X, np.nan)
                for p, ts in trials.items():
                    X_nan[p, ts] = X[p, ts]
                X_noise = X_nan.copy(); mask = np.isnan(X_noise); X_noise[mask] = rng.normal(0, 1e3, mask.sum())
                t0 = time.perf_counter()
                scaler, model = fit_design(X_nan, A, y, people, trials, SEED + fold); FITS['pilot_logistic_fits'] += 1
                t1 = time.perf_counter()
                scaler2, model2 = fit_design(X_noise, A, y, people, trials, SEED + fold); FITS['exclusion_proof_duplicate_fits'] += 1
                p = predict(scaler, model, X, A, te); p2 = predict(scaler2, model2, X, A, te)
                assert np.array_equal(model.coef_, model2.coef_) and np.array_equal(p, p2), 'unacquired data influenced the fit'
                acquired_hash = hashlib.sha256(json.dumps({str(p_): trials[p_].tolist() for p_ in sorted(trials)}).encode()).hexdigest()
                np.savez(OUT / 'models' / f'f{fold}_d{draw}_{design}.npz', coef=model.coef_, intercept=model.intercept_,
                         scale_mean=scaler.mean_, scale_scale=scaler.scale_, classes=model.classes_)
                timings.append({'fold': fold, 'draw': draw, 'design': design, 'fit_seconds': t1 - t0})
                memberships.extend({'fold': fold, 'draw': draw, 'design': design, 'role': 'train', 'participant_id': int(p_) + 1,
                                    'retained_trials': ','.join(str(t + 1) for t in trials[p_]), 'class': int(y[p_])} for p_ in sorted(trials))
                memberships.extend({'fold': fold, 'draw': draw, 'design': design, 'role': 'test', 'participant_id': int(p_) + 1,
                                    'retained_trials': '1,2,3,4,5', 'class': int(y[p_])} for p_ in te)
                predictions.extend({'fold': fold, 'draw': draw, 'design': design, 'participant_id': int(p_) + 1, 'label': int(y[p_]),
                                    'p0': p[j, 0], 'p1': p[j, 1], 'p2': p[j, 2]} for j, p_ in enumerate(te))
                audits.append({'fold': fold, 'draw': draw, 'design': design, 'n': n, 'k': k, 'recordings': n * k,
                               'class_counts': '/'.join(map(str, np.bincount(y[people], minlength=3))), 'acquired_index_sha256': acquired_hash,
                               'exclusion_proof_identical': True, 'nan_in_features': False})
            # Nesting audit
            assert set(people21) <= set(people35)
            for p_ in people21:
                assert set(trials3[int(p_)]) <= set(range(5))
    mem = pd.DataFrame(memberships); mem.to_csv(OUT / 'acquisition_memberships.csv', index=False)
    pred = pd.DataFrame(predictions); pred.to_csv(OUT / 'heldout_predictions.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'acquisition_audit.csv', index=False)
    pd.DataFrame(timings).to_csv(OUT / 'fit_timings.csv', index=False)
    # Descriptive pilot metrics (mechanical; not a basis for design change).
    rows = []
    for (fold, draw, design), g in pred.groupby(['fold', 'draw', 'design']):
        probs = g[['p0', 'p1', 'p2']].to_numpy()
        rows.append({'fold': fold, 'draw': draw, 'design': design, 'balanced_accuracy': balanced_accuracy_score(g.label, probs.argmax(1)),
                     'log_loss': log_loss(g.label, probs, labels=[0, 1, 2])})
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT / 'pilot_metrics_by_fold_draw.csv', index=False)
    paired = metrics.pivot_table(index=['fold', 'draw'], columns='design', values='balanced_accuracy')
    summary = {'design_mean_balanced_accuracy': metrics.groupby('design').balanced_accuracy.mean().round(4).to_dict(),
               'design_mean_log_loss': metrics.groupby('design').log_loss.mean().round(4).to_dict(),
               'paired_mean_difference_A_minus_B': round(float((paired['A_35x3'] - paired['B_21x5']).mean()), 4),
               'paired_pairs': int(len(paired)), 'label': 'MECHANICAL PILOT: repeat 0 only, 5 draws, fixed C=0.1; not a basis for design change'}
    dump('pilot_summary.json', summary)
    dump('progress.json', {**FITS, 'total_fits_executed': sum(FITS.values()), 'authorized_primary_fits': 50,
                           'runtime_seconds': round(sum(t['fit_seconds'] for t in timings), 3),
                           'mean_fit_seconds': round(float(np.mean([t['fit_seconds'] for t in timings])), 4),
                           'full_grid_run': False})
    manifest = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    dump('output_manifest.json', manifest)
    print(json.dumps({**summary, **FITS}, indent=2))


if __name__ == '__main__':
    run()
