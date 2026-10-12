"""Addendum A2: inner-tuning sensitivity on cells 35x3 and 21x5 using grid_v1 memberships. 5,400 inner fits + 600 refits + 600 exclusion controls."""
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
GRID = STUDY / 'grid_v1'
OUT = STUDY / 'sensitivity_v1'
ADDENDUM = STUDY / 'ADDENDUM_A2_SENSITIVITY_TUNING.md'
CACHE = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz'
EXPECTED = {CACHE: '1cdef1a7c3dc181f8e8ccfcdd1585992017d163ff79f6afa785b0e3cae37c924',
            STUDY / 'PROTOCOL_v2_FROZEN.md': 'd98ee18951e3d0fe3e3f62f09d459211890570b31d757e9283035826b6f5a623',
            STUDY / 'ADDENDUM_A1_FULL_GRID.md': '1e43e2c0da3c6b2265a063b997dc947503dd27f52b3a611c486f7455b7ceebae'}
SEED_A2 = 20261103
CS = [0.1, 1.0, 10.0]
CELLS = [(35, 3), (21, 5)]
DRAWS = 20
BOOT = 5000
MAX_ITER = 3000
FITS = {'inner_fits': 0, 'refits': 0, 'exclusion_control_fits': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def feats_from(X_acq, A, people, trials):
    f = np.stack([np.concatenate([np.median(X_acq[p, trials[p]], axis=0), A[p]]) for p in people])
    assert np.isfinite(f).all(), 'NaN reached features'
    return f


def fit(feats, labels, C, seed):
    scaler = StandardScaler().fit(feats)
    model = LogisticRegression(C=C, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(scaler.transform(feats), labels)
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert model.classes_.tolist() == [0, 1, 2]
    return scaler, model, warned, int(np.max(model.n_iter_))


def run():
    if OUT.exists():
        raise FileExistsError(f'Existing outputs preserved: {OUT}')
    for p, h in EXPECTED.items():
        assert sha(p) == h, p
    grid_manifest = json.loads((GRID / 'output_manifest.json').read_text())
    assert sha(GRID / 'acquisition_memberships.csv') == grid_manifest['acquisition_memberships.csv']
    assert sha(GRID / 'heldout_predictions.csv') == grid_manifest['heldout_predictions.csv']
    OUT.mkdir(); (OUT / 'models').mkdir()
    frozen = {'status': 'FROZEN_A2_SENSITIVITY', 'date': datetime.now().isoformat(timespec='seconds'), 'addendum_A2_sha256': sha(ADDENDUM),
              'script_sha256': sha(Path(__file__)), 'input_sha256': {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
              'grid_memberships_sha256': grid_manifest['acquisition_memberships.csv'], 'grid_predictions_sha256': grid_manifest['heldout_predictions.csv'],
              'cells': CELLS, 'C_grid': CS, 'selection_rule': 'argmin mean inner-validation log loss over 3 inner folds; ties -> smaller C',
              'inner_scaler': 'fitted on inner-training people of that inner fold only', 'seed': SEED_A2, 'bootstrap': BOOT,
              'expected_fits': {'inner': 5400, 'refits': 600, 'exclusion_control': 600}, 'gaitrec_authorized': False,
              'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
    (STUDY / 'frozen_protocol_A2.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8'); dump('frozen_protocol_A2_copy.json', frozen)

    cache = np.load(CACHE); X = cache['X']; A = cache['A']; y = cache['labels'].astype(int)
    mem = pd.read_csv(GRID / 'acquisition_memberships.csv'); test_mem = mem[mem.role == 'test']
    grid_pred = pd.read_csv(GRID / 'heldout_predictions.csv')
    inner_rows = []; sel_rows = []; preds = []; conv = []; inner_mem = []
    t_start = time.perf_counter()
    for (repeat, fold), tblock in test_mem.groupby(['repeat', 'fold']):
        te = np.sort(tblock.participant_id.to_numpy()) - 1
        print(f'repeat {repeat} fold {fold}', flush=True)
        for draw in range(DRAWS):
            for n, k in CELLS:
                cell = mem[(mem.role == 'train') & (mem.repeat == repeat) & (mem.fold == fold) & (mem.draw == draw) & (mem.n == n) & (mem.k == k)]
                people = cell.participant_id.to_numpy() - 1
                trials = {int(p) - 1: np.array([int(t) - 1 for t in ts.split(',')]) for p, ts in zip(cell.participant_id, cell.retained_trials)}
                assert len(people) == n and all(len(v) == k for v in trials.values()) and not set(people) & set(te)
                X_nan = np.full_like(X, np.nan)
                for p, ts in trials.items():
                    X_nan[p, ts] = X[p, ts]
                feats = feats_from(X_nan, A, people, trials); labels = y[people]
                skf = StratifiedKFold(3, shuffle=True, random_state=SEED_A2 + repeat * 10000 + fold * 100 + draw)
                losses = {C: [] for C in CS}
                for i, (it, iv) in enumerate(skf.split(people, labels)):
                    inner_mem.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'inner_fold': i, 'role': r, 'participant_id': int(people[j]) + 1}
                                     for r, ids in [('inner_train', it), ('inner_val', iv)] for j in ids)
                    for C in CS:
                        sc, mdl, warned, nit = fit(feats[it], labels[it], C, SEED_A2 + fold); FITS['inner_fits'] += 1
                        p = mdl.predict_proba(sc.transform(feats[iv]))
                        ll = log_loss(labels[iv], p, labels=[0, 1, 2]); losses[C].append(ll)
                        inner_rows.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'inner_fold': i, 'C': C, 'inner_val_log_loss': ll})
                        conv.append({'stage': 'inner', 'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'inner_fold': i, 'C': C, 'n_iter': nit,
                                     'reached_max_iter': nit >= MAX_ITER, 'convergence_warning': warned})
                means = {C: float(np.mean(losses[C])) for C in CS}
                best = min(CS, key=lambda C: (means[C], C))
                sel_rows.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, **{f'mean_inner_ll_C{C}': means[C] for C in CS}, 'selected_C': best})
                sc, mdl, warned, nit = fit(feats, labels, best, SEED_A2 + fold); FITS['refits'] += 1
                X_noise = X_nan.copy(); mask = np.isnan(X_noise); X_noise[mask] = np.random.default_rng(SEED_A2 + draw).normal(0, 1e3, mask.sum())
                sc2, mdl2, _, _ = fit(feats_from(X_noise, A, people, trials), labels, best, SEED_A2 + fold); FITS['exclusion_control_fits'] += 1
                tf = np.concatenate([np.median(X[te], axis=1), A[te]], axis=1)
                p = mdl.predict_proba(sc.transform(tf)); p2 = mdl2.predict_proba(sc2.transform(tf))
                assert np.array_equal(mdl.coef_, mdl2.coef_) and np.array_equal(p, p2), 'unacquired data influenced the refit'
                np.savez(OUT / 'models' / f'r{repeat}_f{fold}_d{draw}_n{n}_k{k}.npz', coef=mdl.coef_, intercept=mdl.intercept_, scale_mean=sc.mean_,
                         scale_scale=sc.scale_, classes=mdl.classes_, C=np.array([best]), n_iter=np.array([nit]))
                conv.append({'stage': 'refit', 'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'inner_fold': -1, 'C': best, 'n_iter': nit,
                             'reached_max_iter': nit >= MAX_ITER, 'convergence_warning': warned})
                preds.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'participant_id': int(p_) + 1, 'label': int(y[p_]),
                              'p0': p[j, 0], 'p1': p[j, 1], 'p2': p[j, 2], 'selected_C': best} for j, p_ in enumerate(te))
    wall = time.perf_counter() - t_start
    pd.DataFrame(inner_rows).to_csv(OUT / 'inner_tuning_losses.csv', index=False)
    sel = pd.DataFrame(sel_rows); sel.to_csv(OUT / 'selected_C.csv', index=False)
    pd.DataFrame(inner_mem).to_csv(OUT / 'inner_fold_memberships.csv', index=False)
    pred = pd.DataFrame(preds); pred.to_csv(OUT / 'heldout_predictions_tuned.csv', index=False)
    cv = pd.DataFrame(conv); cv.to_csv(OUT / 'convergence_audit.csv', index=False)
    summary = summarize(pred, grid_pred, sel, y)
    for name, df in summary.items():
        df.to_csv(OUT / f'{name}.csv', index=False)
    dump('progress.json', {**FITS, 'total_fits_executed': sum(FITS.values()), 'wall_seconds': round(wall, 1),
                           'fits_reached_max_iter': int(cv.reached_max_iter.sum()), 'convergence_warnings': int(cv.convergence_warning.sum()),
                           'max_n_iter': int(cv.n_iter.max()), 'gaitrec_run': False})
    manifest = {str(p.relative_to(OUT)).replace('\\', '/'): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    dump('output_manifest.json', manifest)
    print(summary['tuned_contrasts'].round(4).to_string(index=False)); print(summary['tuned_vs_fixed_by_cell'].round(4).to_string(index=False))
    print(summary['selected_C_distribution'].to_string(index=False)); print(json.dumps(FITS))


def _ba(label, hard, w):
    return float(np.mean([w[(label == c) & (hard == c)].sum() / w[label == c].sum() for c in range(3)]))


def _ll(label, probs, w):
    return float(-(w * np.log(np.clip(probs[np.arange(len(label)), label], 1e-15, 1))).sum() / w.sum())


def summarize(pred, grid_pred, sel, y):
    rng = np.random.default_rng(SEED_A2 + 999)
    idx = [np.where(y == c)[0] for c in range(3)]
    W = np.ones((BOOT + 1, 70))
    for b in range(1, BOOT + 1):
        w = np.zeros(70)
        for c in range(3):
            np.add.at(w, rng.choice(idx[c], len(idx[c]), replace=True), 1)
        W[b] = w

    def series(df):
        out = {}
        for (r, d, n, k), g in df.groupby(['repeat', 'draw', 'n', 'k']):
            if (n, k) not in CELLS:
                continue
            g = g.sort_values('participant_id'); assert g.participant_id.tolist() == list(range(1, 71))
            lab = g.label.to_numpy(); pr = g[['p0', 'p1', 'p2']].to_numpy(); hard = pr.argmax(1)
            out[(r, d, n, k)] = (np.array([_ba(lab, hard, W[b]) for b in range(BOOT + 1)]), np.array([_ll(lab, pr, W[b]) for b in range(BOOT + 1)]))
        return out
    tuned = series(pred); fixed = series(grid_pred)
    keys = [(r, d) for r in range(3) for d in range(DRAWS)]

    def stats(stack):
        m = stack.mean(0); point = stack[:, 0].reshape(3, DRAWS)
        return {'mean': m[0], 'ci_low': np.percentile(m[1:], 2.5), 'ci_high': np.percentile(m[1:], 97.5),
                'within_repeat_sd_mean': point.std(axis=1, ddof=1).mean(), 'sd_of_repeat_means': point.mean(1).std(ddof=1),
                'pairs_positive': int((stack[:, 0] > 0).sum())}
    contrasts = []
    for label_, src in [('tuned', tuned), ('fixed_grid_v1', fixed)]:
        for metric, j in [('balanced_accuracy', 0), ('log_loss', 1)]:
            st = np.stack([src[(r, d, 35, 3)][j] - src[(r, d, 21, 5)][j] for r, d in keys])
            contrasts.append({'regime': label_, 'metric': metric, 'contrast': '35x3_minus_21x5', **stats(st)})
    cells = []
    for n, k in CELLS:
        for metric, j in [('balanced_accuracy', 0), ('log_loss', 1)]:
            t = np.stack([tuned[(r, d, n, k)][j] for r, d in keys]); f = np.stack([fixed[(r, d, n, k)][j] for r, d in keys])
            cells.append({'cell': f'{n}x{k}', 'metric': metric, 'tuned_mean': t.mean(0)[0], 'tuned_ci_low': np.percentile(t.mean(0)[1:], 2.5),
                          'tuned_ci_high': np.percentile(t.mean(0)[1:], 97.5), 'fixed_mean': f.mean(0)[0], **{f'tuned_minus_fixed_{a}': b for a, b in stats(t - f).items()}})
    dist = sel.groupby(['n', 'k', 'repeat']).selected_C.value_counts().unstack(fill_value=0).reset_index()
    dist_all = sel.groupby(['n', 'k']).selected_C.value_counts().unstack(fill_value=0).reset_index(); dist_all['repeat'] = 'all'
    return {'tuned_contrasts': pd.DataFrame(contrasts), 'tuned_vs_fixed_by_cell': pd.DataFrame(cells),
            'selected_C_distribution': pd.concat([dist, dist_all], ignore_index=True)}


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if OUT.exists():
            failed = STUDY / f'sensitivity_v1_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; OUT.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(), 'fits_before_failure': FITS}, indent=2))
        raise
