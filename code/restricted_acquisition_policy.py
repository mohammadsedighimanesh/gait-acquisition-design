"""Addendum A4: sequential acquisition policy simulation on real recordings. Usage: python restricted_acquisition_policy.py pilot|grid"""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
ADDENDUM = STUDY / 'ADDENDUM_A4_SEQUENTIAL_POLICY.md'
TASKS = {
    'FPI': {'cache': ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/source_feature_cache.npz',
            'splits': ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv', 'id_col': 'participant_id',
            'budgets': [105, 70], 'warm': 7, 'batch': 7, 'seeds': 20, 'pca': None, 'classes': 3},
    'GaitRec': {'cache': ROOT / 'idea1_2026/gaitrec_trial_stability_pilot_v1/development_bags.npz',
                'splits': ROOT / 'idea1_2026/gaitrec_trial_stability_pilot_v1/outer_person_splits_before_fit.csv', 'id_col': 'person_id',
                'budgets': [3000, 1500], 'warm': 100, 'batch': 100, 'seeds': 5, 'pca': 12, 'classes': 5}}
EXPECTED = {'FPI': {'cache': '1cdef1a7c3dc181f8e8ccfcdd1585992017d163ff79f6afa785b0e3cae37c924', 'splits': '42b7d70a85da97ca6240b3fa4b85451cec5b13dae97ea1e6588eb463ffdd025c'},
            'GaitRec': {'cache': 'e487d759e58a28333568ede6f8a41453b90a68201499bdaf39e2818e05b58149', 'splits': '15fd4d4d54ca1eb588db3cbd2e3e4d5a5ebc46893e5841e87556adfa6ca8452e'}}
SEED = 20261105; C_FIXED = 0.1; MAX_ITER = 3000; BOOT = 5000
POLICIES = ['CONV5', 'FIX4', 'FIX3', 'FIX2', 'FIX1', 'RANDOM', 'MARGIN']
FITS = {'internal_fits': 0, 'final_fits': 0, 'exclusion_control_fits': 0}
CONV = {'warnings': 0, 'max_n_iter': 0, 'missing_class_fits': 0}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Task:
    def __init__(self, name):
        cfg = TASKS[name]; self.name = name; self.cfg = cfg
        assert sha(cfg['cache']) == EXPECTED[name]['cache'] and sha(cfg['splits']) == EXPECTED[name]['splits']
        d = np.load(cfg['cache'])
        if name == 'FPI':
            self.X = d['X']; self.A = d['A']; self.y = d['labels'].astype(int); self.ids = np.arange(1, 71)
        else:
            self.X = d['X']; self.A = None; self.y = d['y'].astype(int); self.ids = d['person_ids']
        self.pos = {int(i): j for j, i in enumerate(self.ids)}
        self.splits = pd.read_csv(cfg['splits']); self.nc = cfg['classes']

    def person_vec(self, X_src, p, trials):
        v = np.median(X_src[p, trials], axis=0)
        return v if self.A is None else np.concatenate([v, self.A[p]])

    def feats(self, X_src, acquired):
        f = np.stack([self.person_vec(X_src, p, t) for p, t in acquired.items()]); assert np.isfinite(f).all(); return f

    def test_feats(self, te):
        return np.stack([self.person_vec(self.X, p, np.arange(5)) for p in te])

    def fit(self, feats, labels, seed, count_key):
        s1 = StandardScaler().fit(feats); z = s1.transform(feats); pca = None
        if self.cfg['pca']:
            pca = PCA(self.cfg['pca'], svd_solver='full', random_state=seed).fit(z); z = pca.transform(z)
            s2 = StandardScaler().fit(z); z = s2.transform(z)
        else:
            s2 = None
        model = LogisticRegression(C=C_FIXED, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning); model.fit(z, labels)
            CONV['warnings'] += any(issubclass(w.category, ConvergenceWarning) for w in caught)
        CONV['max_n_iter'] = max(CONV['max_n_iter'], int(np.max(model.n_iter_)))
        # A4.2: an acquired set may lack a class (class-blind recruitment at small budgets). The model then cannot predict that class;
        # its probability is 0 (clipped to 1e-15 in log loss) and the class counts as unrecognised. Flagged by `missing_classes`.
        assert set(model.classes_.tolist()) <= set(range(self.nc)); FITS[count_key] += 1
        CONV['missing_class_fits'] = CONV.get('missing_class_fits', 0) + int(len(model.classes_) < self.nc)
        return (s1, pca, s2, model)

    def predict(self, pipe, feats):
        s1, pca, s2, model = pipe; z = s1.transform(feats)
        if pca is not None:
            z = s2.transform(pca.transform(z))
        p = model.predict_proba(z)
        if len(model.classes_) < self.nc:
            full = np.zeros((len(feats), self.nc)); full[:, model.classes_] = p; return full
        return p


def ba(label, hard, nc):
    """Balanced accuracy over the classes PRESENT in `label` (A4.1 amendment: small inner-validation folds may lack a class; absent classes are skipped instead of producing NaN)."""
    present = [c for c in range(nc) if (label == c).any()]
    return float(np.mean([(hard[label == c] == c).mean() for c in present]))


def internal_gains(task, acquired, labels_of, rng, nc):
    """Backward-difference marginal gain per recording for people vs trials, from acquired data only. Returns 3 replicates."""
    people = np.array(list(acquired)); labels = np.array([labels_of[p] for p in people]); reps = []
    warnings.filterwarnings('ignore', category=RuntimeWarning); warnings.filterwarnings('ignore', category=UserWarning)
    for rep in range(3):
        skf = StratifiedKFold(3, shuffle=True, random_state=int(rng.integers(1 << 31)))
        res = {'full': [], 'people': [], 'trial': []}; removed = {'people': [], 'trial': []}
        for it, iv in skf.split(people, labels):
            val = {int(people[j]): acquired[int(people[j])] for j in iv}; vf = task.feats(task.X, val); vl = labels[iv]
            tr_full = {int(people[j]): acquired[int(people[j])] for j in it}
            keep = rng.permutation(it)[:max(nc, int(round(0.8 * len(it))))]
            tr_people = {int(people[j]): acquired[int(people[j])] for j in keep}
            tr_trial = {}
            for j in it:
                p = int(people[j]); t = acquired[p]
                tr_trial[p] = np.sort(np.delete(t, rng.integers(len(t)))) if len(t) >= 2 else t
            variants = {'full': tr_full, 'people': tr_people, 'trial': tr_trial}
            for key, tr in variants.items():
                tl = np.array([labels_of[p] for p in tr])
                if len(set(tl)) < nc:
                    res[key].append(np.nan); continue
                pipe = task.fit(task.feats(task.X, tr), tl, SEED + rep, 'internal_fits')
                res[key].append(ba(vl, task.predict(pipe, vf).argmax(1), nc))
            removed['people'].append(sum(len(t) for t in tr_full.values()) - sum(len(t) for t in tr_people.values()))
            removed['trial'].append(sum(len(t) for t in tr_full.values()) - sum(len(t) for t in tr_trial.values()))
        full = np.nanmean(res['full']); gp = (full - np.nanmean(res['people'])) / max(1, np.mean(removed['people']))
        gt = (full - np.nanmean(res['trial'])) / max(1, np.mean(removed['trial']))
        reps.append({'replicate': rep, 'ba_full': full, 'ba_people_reduced': np.nanmean(res['people']), 'ba_trial_reduced': np.nanmean(res['trial']),
                     'removed_people_recordings': float(np.mean(removed['people'])), 'removed_trial_recordings': float(np.mean(removed['trial'])),
                     'gain_people': gp, 'gain_trial': gt})
    return reps


def run_policy(task, policy, pool_order, trial_order, labels_of, B, rng, log):
    """Return acquired dict {person_index: np.array(trial indices)} with exactly B recordings."""
    cfg = task.cfg; acquired = {}; recruited = []; n_rec = 0; nc = task.nc
    def recruit(k):
        nonlocal n_rec
        for p in pool_order:
            if p in acquired:
                continue
            kk = min(k, B - n_rec)
            if kk <= 0:
                return False
            acquired[p] = np.array(trial_order[p][:kk]); recruited.append(p); n_rec += kk; return True
        return False
    def add_trials(count):
        nonlocal n_rec
        added = 0
        for p in sorted(recruited, key=lambda q: (len(acquired[q]), recruited.index(q))):
            if added >= count or n_rec >= B:
                break
            if len(acquired[p]) < 5:
                acquired[p] = np.array(trial_order[p][:len(acquired[p]) + 1]); n_rec += 1; added += 1
        return added
    if policy.startswith('FIX') or policy == 'CONV5':
        k = 5 if policy == 'CONV5' else int(policy[3])
        while n_rec < B and recruit(k):
            pass
        while n_rec < B:  # pool exhausted: round-robin extra trials
            if add_trials(B - n_rec) == 0:
                raise RuntimeError('budget exceeds pool capacity')
    else:
        for _ in range(cfg['warm']):
            recruit(2)
        step = 0
        while n_rec < B:
            b = min(cfg['batch'], B - n_rec)
            if policy == 'RANDOM':
                action = 'people' if rng.random() < 0.5 else 'trials'; est = None
            else:
                reps = internal_gains(task, acquired, labels_of, rng, nc); chosen = int(rng.integers(3)); est = reps[chosen]
                if np.isnan(est['gain_people']) or np.isnan(est['gain_trial']):
                    action = 'people'  # A4.1: internal estimate undefined (a class has too few acquired people); only recruiting can make it defined
                    for r in reps:
                        r['undefined_estimate'] = True
                else:
                    action = 'people' if est['gain_people'] >= est['gain_trial'] else 'trials'
                    for r in reps:
                        r['undefined_estimate'] = False
                for r in reps:
                    log.append({'step': step, 'n_people': len(acquired), 'recordings': n_rec, **r, 'chosen_replicate': chosen, 'action': action})
            if action == 'people':
                got = 0
                while got < b and n_rec < B and recruit(1):
                    got += 1
                if got < b and n_rec < B:
                    add_trials(b - got)
            else:
                added = add_trials(b)
                if added < b:
                    while added < b and n_rec < B and recruit(1):
                        added += 1
            if policy == 'RANDOM':
                log.append({'step': step, 'n_people': len(acquired), 'recordings': n_rec, 'action': action})
            step += 1
    assert sum(len(t) for t in acquired.values()) == B and all(len(set(t)) == len(t) <= 5 for t in acquired.values())
    return acquired


def summarize(pred, y_all, nc, policies, budgets):
    rows = []; keys = sorted(set(zip(pred.repeat, pred.fold, pred.seed)))
    repeats = sorted(pred.repeat.unique())
    rng = np.random.default_rng(SEED + 999); idx = [np.where(y_all == c)[0] for c in range(nc)]
    W = np.ones((BOOT + 1, len(y_all)))
    for b in range(1, BOOT + 1):
        w = np.zeros(len(y_all))
        for c in range(nc):
            np.add.at(w, rng.choice(idx[c], len(idx[c]), replace=True), 1)
        W[b] = w
    # pooled per (repeat, seed, budget, policy) over folds -> all people once
    metrics = {}
    for (r, s, B, pol), g in pred.groupby(['repeat', 'seed', 'budget', 'policy']):
        g = g.sort_values('person_index'); assert g.person_index.tolist() == list(range(len(y_all)))
        lab = g.label.to_numpy(); probs = g[[f'p{c}' for c in range(nc)]].to_numpy(); hard = probs.argmax(1)
        bav = np.zeros(BOOT + 1)
        for c in range(nc):
            inc = (lab == c).astype(float); cor = ((lab == c) & (hard == c)).astype(float); bav += (W @ cor) / (W @ inc)
        ll = -(W @ np.log(np.clip(probs[np.arange(len(lab)), lab], 1e-15, 1))) / W.sum(1)
        metrics[(r, s, B, pol)] = (bav / nc, ll)
    seeds = sorted(pred.seed.unique())
    def stats(stack):
        m = stack.mean(0); point = stack[:, 0].reshape(len(repeats), len(seeds))
        return {'mean': m[0], 'ci_low': np.percentile(m[1:], 2.5), 'ci_high': np.percentile(m[1:], 97.5),
                'within_repeat_sd_mean': point.std(axis=1, ddof=1).mean() if len(seeds) > 1 else float('nan'),
                'sd_of_repeat_means': point.mean(1).std(ddof=1) if len(repeats) > 1 else float('nan'), 'pairs_positive': int((stack[:, 0] > 0).sum()), 'pairs': stack.shape[0]}
    out = []
    for B in budgets:
        for pol in policies:
            for mi, mname in enumerate(['balanced_accuracy', 'log_loss']):
                st = np.stack([metrics[(r, s, B, pol)][mi] for r in repeats for s in seeds]); sb = stats(st)
                out.append({'budget': B, 'policy': pol, 'metric': mname, **{k: v for k, v in sb.items() if k != 'pairs_positive'}})
        fixed = [p for p in policies if p.startswith('FIX') or p == 'CONV5']
        oracle = max(fixed, key=lambda p: np.mean([metrics[(r, s, B, p)][0][0] for r in repeats for s in seeds]))
        for ref in ['CONV5', 'RANDOM', oracle]:
            for mi, mname in enumerate(['balanced_accuracy', 'log_loss']):
                st = np.stack([metrics[(r, s, B, 'MARGIN')][mi] - metrics[(r, s, B, ref)][mi] for r in repeats for s in seeds])
                out.append({'budget': B, 'policy': f'MARGIN_minus_{ref}' + ('_ORACLEFIX' if ref == oracle and ref not in ('CONV5', 'RANDOM') else ''), 'metric': mname, **stats(st)})
    return pd.DataFrame(out)


def run(stage):
    out = STUDY / f'policy_{stage}_v1'
    if out.exists():
        raise FileExistsError(f'Existing outputs preserved: {out}')
    out.mkdir(); (out / 'models').mkdir()
    frozen_path = STUDY / 'frozen_protocol_A4.json'
    if stage == 'pilot':
        frozen = {'status': 'FROZEN_A4_POLICY', 'date': datetime.now().isoformat(timespec='seconds'), 'addendum_A4_sha256': sha(ADDENDUM), 'script_sha256': sha(Path(__file__)),
                  'inputs': EXPECTED, 'tasks': {k: {kk: vv for kk, vv in v.items() if kk not in ('cache', 'splits')} for k, v in TASKS.items()},
                  'policies': POLICIES, 'C': C_FIXED, 'seed': SEED, 'bootstrap': BOOT, 'hyperparameter_tuning': 'none',
                  'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    assert frozen['script_sha256'] == sha(Path(__file__)) and frozen['addendum_A4_sha256'] == sha(ADDENDUM), 'changed after freeze'
    (out / 'frozen_protocol_A4_copy.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    plan = {'pilot': [('FPI', [0], 2, [105])], 'grid': [('FPI', [0, 1, 2], 20, [105, 70]), ('GaitRec', [0, 1, 2], 5, [3000, 1500])]}[stage]
    t_all = time.perf_counter(); summaries = []
    for tname, repeats, nseeds, budgets in plan:
        task = Task(tname); acq_rows = []; pred_rows = []; dec_rows = []; labels_of = {int(i): int(c) for i, c in enumerate(task.y)}
        sp = task.splits[task.splits.repeat.isin(repeats)]
        for (repeat, fold), block in sp.groupby(['repeat', 'fold']):
            tr = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'train', task.cfg['id_col']]))
            te = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'test', task.cfg['id_col']]))
            assert not set(tr) & set(te); tf = task.test_feats(te); tl = task.y[te]
            for seed in range(nseeds):
                print(f'{stage} {tname} r{repeat} f{fold} seed{seed}', flush=True)
                crn = np.random.default_rng(SEED + repeat * 100000 + fold * 1000 + seed)
                pool_order = [int(p) for p in crn.permutation(tr)]; trial_order = {int(p): crn.permutation(5) for p in tr}
                for B in budgets:
                    for pol in POLICIES:
                        prng = np.random.default_rng(SEED + repeat * 100000 + fold * 1000 + seed * 10 + budgets.index(B) * 7 + POLICIES.index(pol))
                        log = []; acquired = run_policy(task, pol, pool_order, trial_order, labels_of, B, prng, log)
                        for row in log:
                            dec_rows.append({'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, **row})
                        labels = np.array([labels_of[p] for p in acquired])
                        X_nan = np.full_like(task.X, np.nan)
                        for p, t in acquired.items():
                            X_nan[p, t] = task.X[p, t]
                        X_noise = X_nan.copy(); m = np.isnan(X_noise); X_noise[m] = prng.normal(0, 1e3, m.sum())
                        pipe = task.fit(task.feats(X_nan, acquired), labels, SEED + fold, 'final_fits')
                        pipe2 = task.fit(task.feats(X_noise, acquired), labels, SEED + fold, 'exclusion_control_fits')
                        p = task.predict(pipe, tf); p2 = task.predict(pipe2, tf)
                        assert np.array_equal(pipe[3].coef_, pipe2[3].coef_) and np.array_equal(p, p2), 'unacquired data influenced the final fit'
                        s1, pca, s2, model = pipe; missing = int(model.classes_.shape[0] < task.nc)
                        np.savez(out / 'models' / f'{tname}_r{repeat}_f{fold}_s{seed}_B{B}_{pol}.npz', s1_mean=s1.mean_, s1_scale=s1.scale_, coef=model.coef_, intercept=model.intercept_,
                                 classes=model.classes_, **({'pca_mean': pca.mean_, 'pca_components': pca.components_, 's2_mean': s2.mean_, 's2_scale': s2.scale_} if pca is not None else {}))
                        acq_rows.extend({'task': tname, 'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'person_id': int(task.ids[pp]),
                                         'trials': ','.join(str(t + 1) for t in np.sort(tt)), 'n_trials': len(tt), 'class': labels_of[pp], 'model_missing_class': missing} for pp, tt in acquired.items())
                        pred_rows.extend({'task': tname, 'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'label': int(tl[j]),
                                          **{f'p{c}': p[j, c] for c in range(task.nc)}} for j, pp in enumerate(te))
        acq = pd.DataFrame(acq_rows); acq.to_csv(out / f'{tname}_acquisitions.csv', index=False)
        pred = pd.DataFrame(pred_rows); pred.to_csv(out / f'{tname}_heldout_predictions.csv', index=False)
        pd.DataFrame(dec_rows).to_csv(out / f'{tname}_policy_decisions.csv', index=False)
        summ = summarize(pred, task.y, task.nc, POLICIES, budgets); summ.insert(0, 'task', tname); summ.to_csv(out / f'{tname}_policy_summary.csv', index=False); summaries.append(summ)
        comp = acq.groupby(['budget', 'policy']).agg(mean_people=('person_index', 'nunique'), mean_trials_per_person=('n_trials', 'mean')).reset_index()
        comp = acq.groupby(['task', 'repeat', 'fold', 'seed', 'budget', 'policy']).agg(people=('person_index', 'nunique'), trials_per_person=('n_trials', 'mean')).reset_index() \
            .groupby(['task', 'budget', 'policy']).agg(mean_people=('people', 'mean'), sd_people=('people', 'std'), mean_trials_per_person=('trials_per_person', 'mean')).reset_index()
        comp.to_csv(out / f'{tname}_acquired_composition.csv', index=False); print(comp.round(3).to_string(index=False))
        print(summ[summ.metric == 'balanced_accuracy'].round(4).to_string(index=False))
    (out / 'progress.json').write_text(json.dumps({**FITS, 'total_fits_executed': sum(FITS.values()), 'stage': stage, 'wall_seconds': round(time.perf_counter() - t_all, 1),
                                                   'convergence_warnings': CONV['warnings'], 'max_n_iter': CONV['max_n_iter'], 'missing_class_fits': CONV['missing_class_fits'], 'official_test_used': False}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2)); print(json.dumps(FITS))


if __name__ == '__main__':
    stage = sys.argv[1]; assert stage in ('pilot', 'grid')
    try:
        run(stage)
    except Exception as exc:
        out = STUDY / f'policy_{stage}_v1'
        if out.exists():
            failed = STUDY / f'policy_{stage}_v1_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; out.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(), 'fits_before_failure': FITS}, indent=2))
        raise
