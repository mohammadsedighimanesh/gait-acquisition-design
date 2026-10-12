"""Addendum A3: GaitRec restricted-acquisition learning surface. Usage: python restricted_acquisition_gaitrec.py pilot|grid"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
SRC = ROOT / 'idea1_2026/gaitrec_trial_stability_pilot_v1'
ADDENDUM = STUDY / 'ADDENDUM_A3_GAITREC.md'
BAGS = SRC / 'development_bags.npz'; SPLITS = SRC / 'outer_person_splits_before_fit.csv'; SESSIONS = SRC / 'selected_development_sessions.csv'
META = ROOT / 'idea1_2026/gaitrec_source_vertical_v1/GRF_metadata.csv'
EXPECTED = {BAGS: 'e487d759e58a28333568ede6f8a41453b90a68201499bdaf39e2818e05b58149',
            SPLITS: '15fd4d4d54ca1eb588db3cbd2e3e4d5a5ebc46893e5841e87556adfa6ca8452e',
            SESSIONS: 'ff33d4005ca5df90cd1f0b113b152e4836ba692f37a339acbb8981f1297f1b58',
            META: '0d89799a2703559a836437560f2a05fbc328f6f6aea1cb73d44ee71b9952191d'}
SEED = 20261104
C_FIXED = 0.1; MAX_ITER = 3000; PCA_K = 12; DRAWS = 20; BOOT = 5000
NS = [300, 500, 600, 800, 1000]; KS = [1, 3, 5]
ALLOC = {300: (33, 54, 85, 81, 47), 500: (56, 90, 141, 135, 78), 600: (67, 108, 170, 161, 94), 800: (90, 144, 226, 215, 125), 1000: (112, 180, 283, 269, 156)}
CONTRASTS = [('primary_1000x3_minus_600x5', (1000, 3), (600, 5)), ('secondary_500x3_minus_300x5', (500, 3), (300, 5))]
FITS = {'primary_fits': 0, 'exclusion_control_fits': 0}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def feats_from(X_acq, people, trials):
    f = np.stack([np.median(X_acq[p, trials[int(p)]], axis=0) for p in people])
    assert np.isfinite(f).all(), 'NaN reached features'
    return f


def fit_pipeline(feats, labels, seed):
    s1 = StandardScaler().fit(feats); z = s1.transform(feats)
    pca = PCA(PCA_K, svd_solver='full', random_state=seed).fit(z); z = pca.transform(z)
    s2 = StandardScaler().fit(z); z = s2.transform(z)
    model = LogisticRegression(C=C_FIXED, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning); model.fit(z, labels)
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert model.classes_.tolist() == [0, 1, 2, 3, 4]
    return (s1, pca, s2, model), warned, int(np.max(model.n_iter_))


def predict(pipe, feats):
    s1, pca, s2, model = pipe
    return model.predict_proba(s2.transform(pca.transform(s1.transform(feats))))


def weighted_metrics(label, probs, W):
    """Vectorised: W is (B,P) weights. Returns BA (B,) and log loss (B,)."""
    hard = probs.argmax(1); ba = np.zeros(W.shape[0])
    for c in range(5):
        inc = (label == c).astype(float); cor = ((label == c) & (hard == c)).astype(float)
        ba += (W @ cor) / (W @ inc)
    ll = -(W @ np.log(np.clip(probs[np.arange(len(label)), label], 1e-15, 1))) / W.sum(1)
    return ba / 5, ll


def make_weights(y, boot, seed):
    rng = np.random.default_rng(seed); W = np.ones((boot + 1, len(y)))
    idx = [np.where(y == c)[0] for c in range(5)]
    for b in range(1, boot + 1):
        w = np.zeros(len(y))
        for c in range(5):
            np.add.at(w, rng.choice(idx[c], len(idx[c]), replace=True), 1)
        W[b] = w
    return W


def summarize(pred, y, cells, contrasts, draws):
    ns = sorted({n for n, _ in cells}); ks = sorted({k for _, k in cells})
    W = make_weights(y, BOOT, SEED + 999)
    groups = {}
    for (r, d, n, k), g in pred.groupby(['repeat', 'draw', 'n', 'k']):
        g = g.sort_values('person_index'); assert g.person_index.tolist() == list(range(len(y)))
        groups[(r, d, n, k)] = weighted_metrics(g.label.to_numpy(), g[[f'p{c}' for c in range(5)]].to_numpy(), W)
    repeats = sorted(pred.repeat.unique())

    def stats(stack):
        m = stack.mean(0); point = stack[:, 0].reshape(len(repeats), draws)
        return {'mean': m[0], 'ci_low': np.percentile(m[1:], 2.5), 'ci_high': np.percentile(m[1:], 97.5),
                'within_repeat_sd_mean': point.std(axis=1, ddof=1).mean() if draws > 1 else float('nan'),
                'sd_of_repeat_means': point.mean(1).std(ddof=1) if len(repeats) > 1 else float('nan'), 'pairs_positive': int((stack[:, 0] > 0).sum())}
    keys = [(r, d) for r in repeats for d in range(draws)]
    surface = []
    for n, k in cells:
        if True:
            ba = np.stack([groups[(r, d, n, k)][0] for r, d in keys]); ll = np.stack([groups[(r, d, n, k)][1] for r, d in keys])
            sb = stats(ba); sl = stats(ll)
            surface.append({'n': n, 'k': k, 'recordings': n * k, 'balanced_accuracy': sb['mean'], 'ba_ci_low': sb['ci_low'], 'ba_ci_high': sb['ci_high'],
                            'ba_within_repeat_sd_mean': sb['within_repeat_sd_mean'], 'ba_sd_of_repeat_means': sb['sd_of_repeat_means'],
                            'log_loss': sl['mean'], 'll_ci_low': sl['ci_low'], 'll_ci_high': sl['ci_high'], 'n_values_averaged': len(keys)})
    con = []
    for name, (na, ka), (nb, kb) in contrasts:
        if (na, ka) in cells and (nb, kb) in cells:
            for metric, j in [('balanced_accuracy', 0), ('log_loss', 1)]:
                st = np.stack([groups[(r, d, na, ka)][j] - groups[(r, d, nb, kb)][j] for r, d in keys])
                con.append({'contrast': name, 'metric': metric, 'recordings_A': na * ka, 'recordings_B': nb * kb, 'paired_pairs': len(keys), **stats(st)})
    fin = []
    for n, k in cells:
        for n2, k2 in cells:
            if (n2 == n and ks.index(k2) == ks.index(k) + 1) or (k2 == k and ns.index(n2) == ns.index(n) + 1):
                st = np.stack([groups[(r, d, n2, k2)][0] - groups[(r, d, n, k)][0] for r, d in keys]); s = stats(st)
                fin.append({'from': f'{n}x{k}', 'to': f'{n2}x{k2}', 'axis': 'k' if n2 == n else 'n', 'mean_difference': s['mean'], 'ci_low': s['ci_low'], 'ci_high': s['ci_high']})
    cellrows = [{'repeat': r, 'draw': d, 'n': n, 'k': k, 'balanced_accuracy': v[0][0], 'log_loss': v[1][0]} for (r, d, n, k), v in groups.items()]
    return pd.DataFrame(surface), pd.DataFrame(con), pd.DataFrame(fin), pd.DataFrame(cellrows)


def run(stage):
    out = STUDY / f'gaitrec_{stage}_v1'
    if out.exists():
        raise FileExistsError(f'Existing outputs preserved: {out}')
    for p, h in EXPECTED.items():
        assert sha(p) == h, p
    ns, ks, draws, repeats = (NS, KS, DRAWS, [0, 1, 2]) if stage == 'grid' else ([600, 1000], [3, 5], 5, [0])
    cells = [(1000, 3), (600, 5)] if stage == 'pilot' else [(n, k) for n in ns for k in ks]
    out.mkdir(); (out / 'models').mkdir()
    frozen_path = STUDY / 'frozen_protocol_A3.json'
    if stage == 'pilot':
        frozen = {'status': 'FROZEN_A3_GAITREC', 'date': datetime.now().isoformat(timespec='seconds'), 'addendum_A3_sha256': sha(ADDENDUM),
                  'script_sha256': sha(Path(__file__)), 'input_sha256': {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
                  'grid': {'n': NS, 'k': KS, 'draws': DRAWS}, 'allocation': {str(n): list(v) for n, v in ALLOC.items()}, 'C': C_FIXED, 'pca': PCA_K,
                  'contrasts': [c[0] for c in CONTRASTS], 'seed': SEED, 'bootstrap': BOOT, 'official_test_used': False,
                  'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    assert frozen['script_sha256'] == sha(Path(__file__)) and frozen['addendum_A3_sha256'] == sha(ADDENDUM), 'script or addendum changed after freeze'
    (out / 'frozen_protocol_A3_copy.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')

    bags = np.load(BAGS); X = bags['X']; y = bags['y'].astype(int); pids = bags['person_ids']
    meta = pd.read_csv(META); test_ids = set(meta.loc[meta.TEST == 1, 'SUBJECT_ID']); assert not set(pids) & test_ids, 'official TEST person present'
    pos = {int(p): i for i, p in enumerate(pids)}
    splits = pd.read_csv(SPLITS); splits = splits[splits.repeat.isin(repeats)]
    memberships = []; predictions = []; audits = []; conv = []; t0_all = time.perf_counter()
    for (repeat, fold), block in splits.groupby(['repeat', 'fold']):
        tr = np.array(sorted(pos[int(p)] for p in block.loc[block.role == 'train', 'person_id']))
        te = np.array(sorted(pos[int(p)] for p in block.loc[block.role == 'test', 'person_id']))
        assert not set(tr) & set(te) and len(tr) + len(te) == 1301
        test_feats = np.concatenate([np.median(X[te], axis=1)], axis=1)
        print(f'{stage} repeat {repeat} fold {fold}', flush=True)
        for draw in range(draws):
            rng = np.random.default_rng(SEED + repeat * 10000 + fold * 100 + draw)
            order = {c: rng.permutation(tr[y[tr] == c]) for c in range(5)}
            trial_order = {int(p): rng.permutation(5) for p in tr}
            for n in ns:
                people = np.sort(np.concatenate([order[c][:ALLOC[n][c]] for c in range(5)]))
                for k in ks:
                    if (n, k) not in cells:
                        continue
                    trials = {int(p): np.sort(trial_order[int(p)][:k]) for p in people}
                    assert tuple(np.bincount(y[people], minlength=5)) == ALLOC[n] and len(set(y[people])) == 5 and not set(people) & set(te)
                    X_nan = np.full_like(X, np.nan)
                    for p, ts in trials.items():
                        X_nan[p, ts] = X[p, ts]
                    X_noise = X_nan.copy(); mask = np.isnan(X_noise); X_noise[mask] = rng.normal(0, 1e3, mask.sum())
                    labels = y[people]
                    t0 = time.perf_counter(); pipe, warned, nit = fit_pipeline(feats_from(X_nan, people, trials), labels, SEED + fold); FITS['primary_fits'] += 1
                    t1 = time.perf_counter(); pipe2, _, _ = fit_pipeline(feats_from(X_noise, people, trials), labels, SEED + fold); FITS['exclusion_control_fits'] += 1
                    p = predict(pipe, test_feats); p2 = predict(pipe2, test_feats)
                    identical = bool(np.array_equal(pipe[3].coef_, pipe2[3].coef_) and np.array_equal(p, p2)); assert identical, 'unacquired data influenced the fit'
                    s1, pca, s2, model = pipe
                    np.savez(out / 'models' / f'r{repeat}_f{fold}_d{draw}_n{n}_k{k}.npz', s1_mean=s1.mean_, s1_scale=s1.scale_, pca_mean=pca.mean_, pca_components=pca.components_,
                             s2_mean=s2.mean_, s2_scale=s2.scale_, coef=model.coef_, intercept=model.intercept_, classes=model.classes_, n_iter=np.array([nit]))
                    conv.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'n_iter': nit, 'reached_max_iter': nit >= MAX_ITER, 'convergence_warning': warned, 'fit_seconds': t1 - t0})
                    memberships.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'role': 'train', 'person_index': int(p_), 'person_id': int(pids[p_]),
                                        'retained_trials': ','.join(str(t + 1) for t in trials[int(p_)]), 'class': int(y[p_])} for p_ in people)
                    predictions.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'person_index': int(p_), 'person_id': int(pids[p_]), 'label': int(y[p_]),
                                        **{f'p{c}': p[j, c] for c in range(5)}} for j, p_ in enumerate(te))
                    audits.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'recordings': n * k, 'class_counts': '/'.join(map(str, np.bincount(y[people], minlength=5))),
                                   'exclusion_proof_identical': identical, 'class_order': '0,1,2,3,4'})
        memberships.extend({'repeat': repeat, 'fold': fold, 'draw': -1, 'n': 0, 'k': 0, 'role': 'test', 'person_index': int(p_), 'person_id': int(pids[p_]),
                            'retained_trials': '1,2,3,4,5', 'class': int(y[p_])} for p_ in te)
    wall = time.perf_counter() - t0_all
    pd.DataFrame(memberships).to_csv(out / 'acquisition_memberships.csv', index=False)
    pred = pd.DataFrame(predictions); pred.to_csv(out / 'heldout_predictions.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'acquisition_audit.csv', index=False); cv = pd.DataFrame(conv); cv.to_csv(out / 'convergence_audit.csv', index=False)
    surface, con, fin, cellrows = summarize(pred, y, cells, CONTRASTS if stage == 'grid' else [CONTRASTS[0]], draws)
    surface.to_csv(out / 'learning_surface.csv', index=False); con.to_csv(out / 'prespecified_contrasts.csv', index=False)
    fin.to_csv(out / 'finite_differences.csv', index=False); cellrows.to_csv(out / 'cell_metrics_by_repeat_draw.csv', index=False)
    (out / 'progress.json').write_text(json.dumps({**FITS, 'total_fits_executed': sum(FITS.values()), 'stage': stage, 'wall_seconds': round(wall, 1),
        'mean_primary_fit_seconds': round(float(cv.fit_seconds.mean()), 4), 'fits_reached_max_iter': int(cv.reached_max_iter.sum()),
        'convergence_warnings': int(cv.convergence_warning.sum()), 'max_n_iter': int(cv.n_iter.max()), 'official_test_used': False}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(surface.round(4).to_string(index=False)); print(con.round(4).to_string(index=False)); print(json.dumps(FITS))


if __name__ == '__main__':
    stage = sys.argv[1]; assert stage in ('pilot', 'grid')
    try:
        run(stage)
    except Exception as exc:
        out = STUDY / f'gaitrec_{stage}_v1'
        if out.exists():
            failed = STUDY / f'gaitrec_{stage}_v1_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; out.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(), 'fits_before_failure': FITS}, indent=2))
        raise
