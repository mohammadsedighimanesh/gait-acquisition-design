"""Addendum A5.2: small-budget sequential policy on StepUP. Usage: python restricted_acquisition_stepup_a52.py L1|L2"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import restricted_acquisition_stepup as g

STUDY = g.STUDY; TASKDIR = g.TASKDIR
ADDENDUM = STUDY / 'ADDENDUM_A5_2_SMALL_BUDGET_POLICY.md'
BUDGETS = [150, 100]; WARM = 10; BATCH = 10; SEEDS = 10


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ConstantModel:
    def __init__(self, c, k):
        self.classes_ = np.array([c]); self.coef_ = np.zeros((1, k)); self.intercept_ = np.zeros(1); self.n_iter_ = np.array([0])
    def predict_proba(self, z):
        return np.ones((len(z), 1))


class TaskA52(g.Task):
    """A5.2.1: with tiny acquired sets (FIX10/FIX20 at B<=150 give 5-15 people) PCA(12) is infeasible; use PCA(min(12, n_people-1)) and count such fits."""
    def fit(self, feats, labels, seed, count_key):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        k = min(g.PCA_K, len(feats) - 1)
        if k < g.PCA_K:
            g.CONV['reduced_pca_fits'] = g.CONV.get('reduced_pca_fits', 0) + 1
        s1 = StandardScaler().fit(feats); z = s1.transform(feats); pca = PCA(k, svd_solver='full', random_state=seed).fit(z); z = pca.transform(z); s2 = StandardScaler().fit(z); z = s2.transform(z)
        if len(set(labels.tolist())) < 2:
            # A5.2.2: a single-class acquired set (tiny fixed designs) cannot train a classifier; the model is the constant predictor of that class (A4.2 missing-class rule then applies).
            g.CONV['single_class_fits'] = g.CONV.get('single_class_fits', 0) + 1; g.FITS[count_key] += 1; g.CONV['missing_class_fits'] += 1
            return (s1, pca, s2, ConstantModel(int(labels[0]), k))
        model = LogisticRegression(C=g.C_FIXED, class_weight='balanced', max_iter=g.MAX_ITER, tol=1e-7, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning); model.fit(z, labels)
            g.CONV['warnings'] += any(issubclass(w.category, ConvergenceWarning) for w in caught)
        g.CONV['max_n_iter'] = max(g.CONV['max_n_iter'], int(np.max(model.n_iter_))); g.FITS[count_key] += 1; g.CONV['missing_class_fits'] += int(len(model.classes_) < self.nc)
        return (s1, pca, s2, model)


def run(L):
    out = TASKDIR / f'policy_a52_{L}_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(); (out / 'models').mkdir()
    fp = STUDY / 'frozen_protocol_A5_2.json'
    if not fp.exists():
        fp.write_text(json.dumps({'status': 'FROZEN_A5_2', 'date': datetime.now().isoformat(timespec='seconds'), 'addendum_A5_2_sha256': sha(ADDENDUM), 'script_sha256': sha(Path(__file__)),
                                  'stepup_module_sha256': sha(ROOT / 'restricted_acquisition_stepup.py'), 'features_cache_sha256': sha(TASKDIR / 'build/features_cache.npz'),
                                  'budgets': BUDGETS, 'warm': WARM, 'batch': BATCH, 'seeds': SEEDS, 'policies': g.POLICIES, 'C': g.C_FIXED, 'kmax': g.KMAX}, indent=2))
    fr = json.loads(fp.read_text()); assert fr['script_sha256'] == sha(Path(__file__)) and fr['addendum_A5_2_sha256'] == sha(ADDENDUM) and fr['stepup_module_sha256'] == sha(ROOT / 'restricted_acquisition_stepup.py')
    (out / 'frozen_protocol_A5_2_copy.json').write_text(json.dumps(fr, indent=2))
    g.WARM = WARM; g.BATCH = BATCH  # recorded overrides
    task = TaskA52(L); labels_of = {int(i): int(c) for i, c in enumerate(task.y)}; acq_rows = []; pred_rows = []; dec_rows = []; t0 = time.perf_counter()
    for (repeat, fold), block in task.splits.groupby(['repeat', 'fold']):
        tr = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'train', 'person_id'])); te = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'test', 'person_id']))
        tf = task.test_feats(te); tl = task.y[te]
        for seed in range(SEEDS):
            print(f'a52 {L} r{repeat} f{fold} seed{seed}', flush=True)
            crn = np.random.default_rng(g.SEED + 5200 + repeat * 100000 + fold * 1000 + seed); pool_order = [int(p) for p in crn.permutation(tr)]; trial_order = {int(p): crn.permutation(g.KMAX) for p in tr}
            for B in BUDGETS:
                for pol in g.POLICIES:
                    prng = np.random.default_rng(g.SEED + 5200 + repeat * 100000 + fold * 1000 + seed * 20 + BUDGETS.index(B) * 10 + g.POLICIES.index(pol))
                    log = []; acquired = g.run_policy(task, pol, pool_order, trial_order, labels_of, B, prng, log)
                    dec_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, **row} for row in log]
                    labels = np.array([labels_of[p] for p in acquired]); X_nan = np.full_like(task.X, np.nan)
                    for p, t in acquired.items():
                        X_nan[p, t] = task.X[p, t]
                    X_noise = X_nan.copy(); m = np.isnan(X_noise); X_noise[m] = prng.normal(0, 1e3, m.sum())
                    pipe = task.fit(task.feats(X_nan, acquired), labels, g.SEED + fold, 'final_fits'); pipe2 = task.fit(task.feats(X_noise, acquired), labels, g.SEED + fold, 'exclusion_control_fits')
                    p = task.predict(pipe, tf); p2 = task.predict(pipe2, tf); assert np.array_equal(pipe[3].coef_, pipe2[3].coef_) and np.array_equal(p, p2)
                    g.save_model(out / 'models' / f'r{repeat}_f{fold}_s{seed}_B{B}_{pol}.npz', pipe); missing = int(len(pipe[3].classes_) < task.nc)
                    acq_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'person_id': int(task.ids[pp]), 'trials': ','.join(str(t + 1) for t in np.sort(tt)),
                                  'n_trials': len(tt), 'class': labels_of[pp], 'model_missing_class': missing} for pp, tt in acquired.items()]
                    pred_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'label': int(tl[j]), 'p0': p[j, 0], 'p1': p[j, 1]} for j, pp in enumerate(te)]
    acq = pd.DataFrame(acq_rows); acq.to_csv(out / 'acquisitions.csv', index=False); pred = pd.DataFrame(pred_rows); pred.to_csv(out / 'heldout_predictions.csv', index=False)
    pd.DataFrame(dec_rows).to_csv(out / 'policy_decisions.csv', index=False)
    summ = g.summarize_policy(pred, task.y, task.nc, g.POLICIES, BUDGETS); summ.to_csv(out / 'policy_summary.csv', index=False)
    comp = acq.groupby(['repeat', 'fold', 'seed', 'budget', 'policy']).agg(people=('person_index', 'nunique'), trials_per_person=('n_trials', 'mean')).reset_index() \
        .groupby(['budget', 'policy']).agg(mean_people=('people', 'mean'), sd_people=('people', 'std'), mean_trials_per_person=('trials_per_person', 'mean')).reset_index()
    comp.to_csv(out / 'acquired_composition.csv', index=False)
    (out / 'progress.json').write_text(json.dumps({**g.FITS, **g.CONV, 'stage': 'policy_a52', 'label': L, 'budgets': BUDGETS, 'warm': WARM, 'batch': BATCH, 'wall_seconds': round(time.perf_counter() - t0, 1)}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2)); print(comp.round(2).to_string(index=False)); print(summ[summ.metric == 'balanced_accuracy'].round(4).to_string(index=False)); print(json.dumps(g.FITS))


if __name__ == '__main__':
    L = sys.argv[1]
    try:
        run(L)
    except Exception as exc:
        out = TASKDIR / f'policy_a52_{L}_v1'
        if out.exists():
            failed = TASKDIR / f'policy_a52_{L}_v1_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; out.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(), 'fits_before_failure': g.FITS}, indent=2))
        raise
