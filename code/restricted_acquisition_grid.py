"""Full FPI primary grid for the restricted-acquisition learning surface (Addendum A1). 5,400 primary fits + 5,400 exclusion-control fits."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import hashlib
import json
import platform
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
OUT = STUDY / 'grid_v1'
ADDENDUM = STUDY / 'ADDENDUM_A1_FULL_GRID.md'
PROTOCOL = STUDY / 'PROTOCOL_v2_FROZEN.md'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
SPLITS = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv'
EXPECTED = {CACHE: '1cdef1a7c3dc181f8e8ccfcdd1585992017d163ff79f6afa785b0e3cae37c924',
            SPLITS: '42b7d70a85da97ca6240b3fa4b85451cec5b13dae97ea1e6588eb463ffdd025c',
            PROTOCOL: 'd98ee18951e3d0fe3e3f62f09d459211890570b31d757e9283035826b6f5a623'}
SEED = 20261102
C_FIXED = 0.1
MAX_ITER = 3000
DRAWS = 20
NS = [21, 28, 35, 42, 49, 56]
KS = [1, 3, 5]
ALLOC = {21: (9, 6, 6), 28: (12, 8, 8), 35: (15, 10, 10), 42: (18, 12, 12), 49: (21, 14, 14), 56: (24, 16, 16)}
BOOT = 5000
FITS = {'primary_fits': 0, 'exclusion_control_fits': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def person_features(X_acq, A, people, trials):
    med = np.stack([np.median(X_acq[p, trials[int(p)]], axis=0) for p in people])
    feats = np.concatenate([med, A[people]], axis=1)
    assert np.isfinite(feats).all(), 'NaN reached features'
    return feats


def fit_design(X_acq, A, y, people, trials, seed):
    feats = person_features(X_acq, A, people, trials)
    scaler = StandardScaler().fit(feats)
    model = LogisticRegression(C=C_FIXED, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(scaler.transform(feats), y[people])
        conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert model.classes_.tolist() == [0, 1, 2]
    return scaler, model, conv_warn


def predict(scaler, model, X, A, people):
    feats = np.concatenate([np.median(X[people], axis=1), A[people]], axis=1)
    return model.predict_proba(scaler.transform(feats))


def weighted_ba(label, pred, w):
    return float(np.mean([(w[(label == c) & (pred == c)].sum()) / w[label == c].sum() for c in range(3)]))


def weighted_ll(label, probs, w):
    p = np.clip(probs[np.arange(len(label)), label], 1e-15, 1)
    return float(-(w * np.log(p)).sum() / w.sum())


def summarize(pred, y):
    """Per (repeat, draw, cell): pooled 70-person BA and log loss; bootstrap with shared person weights."""
    pred = pred.sort_values(['repeat', 'draw', 'n', 'k', 'participant_id']).reset_index(drop=True)
    cells = [(n, k) for n in NS for k in KS]
    groups = {}
    for (r, d, n, k), g in pred.groupby(['repeat', 'draw', 'n', 'k']):
        assert len(g) == 70 and g.participant_id.tolist() == list(range(1, 71))
        groups[(r, d, n, k)] = (g.label.to_numpy(), g[['p0', 'p1', 'p2']].to_numpy())
    rng = np.random.default_rng(SEED + 999)
    classes = y
    idx_by_class = [np.where(classes == c)[0] for c in range(3)]
    weights = np.ones((BOOT + 1, 70))
    for b in range(1, BOOT + 1):
        w = np.zeros(70)
        for c in range(3):
            draw = rng.choice(idx_by_class[c], len(idx_by_class[c]), replace=True)
            np.add.at(w, draw, 1)
        weights[b] = w
    ba = {}; ll = {}
    for key, (label, probs) in groups.items():
        hard = probs.argmax(1)
        ba[key] = np.array([weighted_ba(label, hard, weights[b]) for b in range(BOOT + 1)])
        ll[key] = np.array([weighted_ll(label, probs, weights[b]) for b in range(BOOT + 1)])
    rows = []
    for n, k in cells:
        keys = [(r, d, n, k) for r in range(3) for d in range(DRAWS)]
        stack_ba = np.stack([ba[key] for key in keys]); stack_ll = np.stack([ll[key] for key in keys])
        mean_ba = stack_ba.mean(0); mean_ll = stack_ll.mean(0)
        point = stack_ba[:, 0]
        per_repeat = point.reshape(3, DRAWS)
        rows.append({'n': n, 'k': k, 'recordings': n * k, 'balanced_accuracy': mean_ba[0],
                     'ba_ci_low': np.percentile(mean_ba[1:], 2.5), 'ba_ci_high': np.percentile(mean_ba[1:], 97.5),
                     'log_loss': mean_ll[0], 'll_ci_low': np.percentile(mean_ll[1:], 2.5), 'll_ci_high': np.percentile(mean_ll[1:], 97.5),
                     'between_draw_sd_repeat0': per_repeat[0].std(ddof=1), 'between_draw_sd_repeat1': per_repeat[1].std(ddof=1),
                     'between_draw_sd_repeat2': per_repeat[2].std(ddof=1), 'between_draw_sd_pooled': point.std(ddof=1),
                     'between_repeat_sd_of_draw_means': per_repeat.mean(1).std(ddof=1), 'n_values_averaged': len(keys)})
    surface = pd.DataFrame(rows)
    contrasts = []
    for name, (na, ka), (nb, kb) in [('primary_35x3_minus_21x5', (35, 3), (21, 5)), ('secondary_56x3_minus_35x5', (56, 3), (35, 5))]:
        diff = np.stack([ba[(r, d, na, ka)] - ba[(r, d, nb, kb)] for r in range(3) for d in range(DRAWS)])
        md = diff.mean(0)
        contrasts.append({'contrast': name, 'recordings_A': na * ka, 'recordings_B': nb * kb, 'paired_pairs': diff.shape[0],
                          'mean_difference': md[0], 'ci_low': np.percentile(md[1:], 2.5), 'ci_high': np.percentile(md[1:], 97.5),
                          'between_draw_sd_of_difference': diff[:, 0].std(ddof=1), 'fraction_pairs_positive': float((diff[:, 0] > 0).mean())})
    finite = []
    for n, k in cells:
        for n2, k2 in cells:
            if (n2 == n and k2 > k and KS.index(k2) == KS.index(k) + 1) or (k2 == k and n2 > n and NS.index(n2) == NS.index(n) + 1):
                diff = np.stack([ba[(r, d, n2, k2)] - ba[(r, d, n, k)] for r in range(3) for d in range(DRAWS)]); md = diff.mean(0)
                finite.append({'from': f'{n}x{k}', 'to': f'{n2}x{k2}', 'axis': 'k' if n2 == n else 'n', 'mean_difference': md[0],
                               'ci_low': np.percentile(md[1:], 2.5), 'ci_high': np.percentile(md[1:], 97.5)})
    cellrows = [{'repeat': r, 'draw': d, 'n': n, 'k': k, 'balanced_accuracy': ba[(r, d, n, k)][0], 'log_loss': ll[(r, d, n, k)][0]}
                for (r, d, n, k) in groups]
    return surface, pd.DataFrame(contrasts), pd.DataFrame(finite), pd.DataFrame(cellrows)


def run():
    if OUT.exists():
        raise FileExistsError(f'Existing outputs preserved: {OUT}')
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, f'hash mismatch {path}'
    OUT.mkdir(); (OUT / 'models').mkdir()
    frozen = {'status': 'FROZEN_A1_FULL_GRID', 'date': datetime.now().isoformat(timespec='seconds'),
              'protocol_md_sha256': EXPECTED[PROTOCOL], 'addendum_sha256': sha(ADDENDUM), 'grid_script_sha256': sha(Path(__file__)),
              'input_sha256': {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
              'grid': {'n': NS, 'k': KS, 'draws': DRAWS, 'repeats': 3, 'folds': 5}, 'C': C_FIXED, 'max_iter': MAX_ITER, 'bootstrap': BOOT, 'seed': SEED,
              'expected_primary_fits': 15 * len(NS) * len(KS) * DRAWS, 'expected_exclusion_control_fits': 15 * len(NS) * len(KS) * DRAWS,
              'sensitivity_tuning_authorized': False, 'gaitrec_authorized': False,
              'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
    (STUDY / 'frozen_protocol_A1.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    dump('frozen_protocol_A1_copy.json', frozen)

    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    splits = pd.read_csv(SPLITS)
    memberships = []; predictions = []; audits = []; convergence = []
    t_start = time.perf_counter()
    for (repeat, fold), block in splits.groupby(['repeat', 'fold']):
        tr = np.sort(block.loc[block.role == 'train', 'participant_id'].to_numpy()) - 1
        te = np.sort(block.loc[block.role == 'test', 'participant_id'].to_numpy()) - 1
        assert len(tr) == 56 and len(te) == 14 and not set(tr) & set(te) and tuple(np.bincount(y[tr], minlength=3)) == ALLOC[56]
        print(f'repeat {repeat} fold {fold}', flush=True)
        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + repeat * 10000 + fold * 100 + draw)
            order = {c: rng.permutation(tr[y[tr] == c]) for c in range(3)}
            trial_order = {int(p): rng.permutation(5) for p in tr}
            for n in NS:
                people = np.sort(np.concatenate([order[c][:ALLOC[n][c]] for c in range(3)]))
                for k in KS:
                    trials = {int(p): np.sort(trial_order[int(p)][:k]) for p in people}
                    assert tuple(np.bincount(y[people], minlength=3)) == ALLOC[n] and len(set(y[people])) == 3
                    X_nan = np.full_like(X, np.nan)
                    for p, ts in trials.items():
                        X_nan[p, ts] = X[p, ts]
                    X_noise = X_nan.copy(); mask = np.isnan(X_noise); X_noise[mask] = rng.normal(0, 1e3, mask.sum())
                    t0 = time.perf_counter()
                    scaler, model, warned = fit_design(X_nan, A, y, people, trials, SEED + fold); FITS['primary_fits'] += 1
                    t1 = time.perf_counter()
                    scaler2, model2, _ = fit_design(X_noise, A, y, people, trials, SEED + fold); FITS['exclusion_control_fits'] += 1
                    p = predict(scaler, model, X, A, te); p2 = predict(scaler2, model2, X, A, te)
                    identical = bool(np.array_equal(model.coef_, model2.coef_) and np.array_equal(p, p2))
                    assert identical, 'unacquired data influenced the fit'
                    tag = f'r{repeat}_f{fold}_d{draw}_n{n}_k{k}'
                    n_iter = int(np.max(model.n_iter_))
                    np.savez(OUT / 'models' / f'{tag}.npz', coef=model.coef_, intercept=model.intercept_, scale_mean=scaler.mean_,
                             scale_scale=scaler.scale_, classes=model.classes_, n_iter=np.array([n_iter]))
                    convergence.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'n_iter': n_iter,
                                        'reached_max_iter': n_iter >= MAX_ITER, 'convergence_warning': warned, 'fit_seconds': t1 - t0})
                    memberships.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'role': 'train', 'participant_id': int(p_) + 1,
                                        'retained_trials': ','.join(str(t + 1) for t in trials[int(p_)]), 'class': int(y[p_])} for p_ in people)
                    predictions.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'participant_id': int(p_) + 1,
                                        'label': int(y[p_]), 'p0': p[j, 0], 'p1': p[j, 1], 'p2': p[j, 2]} for j, p_ in enumerate(te))
                    audits.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'recordings': n * k,
                                   'class_counts': '/'.join(map(str, np.bincount(y[people], minlength=3))), 'exclusion_proof_identical': identical,
                                   'class_order': '0,1,2'})
        # test memberships once per fold (identical across cells/draws)
        memberships.extend({'repeat': repeat, 'fold': fold, 'draw': -1, 'n': 0, 'k': 0, 'role': 'test', 'participant_id': int(p_) + 1,
                            'retained_trials': '1,2,3,4,5', 'class': int(y[p_])} for p_ in te)
    wall = time.perf_counter() - t_start
    pd.DataFrame(memberships).to_csv(OUT / 'acquisition_memberships.csv', index=False)
    pred = pd.DataFrame(predictions); pred.to_csv(OUT / 'heldout_predictions.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'acquisition_audit.csv', index=False)
    conv = pd.DataFrame(convergence); conv.to_csv(OUT / 'convergence_audit.csv', index=False)
    surface, contrasts, finite, cellrows = summarize(pred, y)
    surface.to_csv(OUT / 'learning_surface.csv', index=False); contrasts.to_csv(OUT / 'prespecified_contrasts.csv', index=False)
    finite.to_csv(OUT / 'finite_differences.csv', index=False); cellrows.to_csv(OUT / 'cell_metrics_by_repeat_draw.csv', index=False)
    dump('progress.json', {**FITS, 'total_fits_executed': sum(FITS.values()), 'expected_primary_fits': 5400,
                           'wall_seconds': round(wall, 1), 'mean_primary_fit_seconds': round(float(conv.fit_seconds.mean()), 4),
                           'fits_reached_max_iter': int(conv.reached_max_iter.sum()), 'convergence_warnings': int(conv.convergence_warning.sum()),
                           'max_n_iter': int(conv.n_iter.max()), 'sensitivity_run': False, 'gaitrec_run': False})
    manifest = {}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'output_manifest.json':
            manifest[str(p.relative_to(OUT)).replace('\\', '/')] = sha(p)
    dump('output_manifest.json', manifest)
    print(surface[['n', 'k', 'balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'between_draw_sd_pooled']].round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False)); print(json.dumps(FITS))


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if OUT.exists():
            failed = STUDY / f'grid_v1_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            OUT.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(),
                                                              'fits_before_failure': FITS}, indent=2), encoding='utf-8')
        raise
