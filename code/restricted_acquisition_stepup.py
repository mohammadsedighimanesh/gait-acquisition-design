"""Addendum A5: StepUP-P150 third task (sex / BMI labels, up to 20 right footsteps per person).
Usage: python restricted_acquisition_stepup.py build | pilot | surface L1|L2 | policy L1|L2 | sensitivity L1
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import hashlib
import io
import json
import platform
import sys
import time
import traceback
import warnings
import zipfile
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
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'
TASKDIR = ROOT / 'idea1_2026/stepup_third_task_v1'
ADDENDUM = STUDY / 'ADDENDUM_A5_STEPUP_THIRD_TASK.md'
SOURCES = [ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources', ROOT / 'idea1_2026/stepup_reliability_extension_v1/sources', TASKDIR / 'sources']
META = ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources/participant_metadata.csv'
CHK = ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources/frdr-dfdr-checksums.txt'
SEED = 20261106; C_FIXED = 0.1; MAX_ITER = 3000; PCA_K = 12; BOOT = 5000; KMAX = 20
NS = [30, 60, 90, 120]; KS = [1, 3, 5, 10, 20]; DRAWS = 20
CONTRASTS = [('primary_120x5_minus_60x10', (120, 5), (60, 10)), ('secondary_60x5_minus_30x10', (60, 5), (30, 10)), ('secondary_60x10_minus_30x20', (60, 10), (30, 20))]
LABELS = {'L1': 'sex', 'L2': 'bmi25'}
POLICIES = ['CONV5', 'FIX1', 'FIX2', 'FIX3', 'FIX4', 'FIX10', 'FIX20', 'RANDOM', 'MARGIN']
POLICY_BUDGETS = [600, 300]; WARM = 10; BATCH = 20; POLICY_SEEDS = 10
FITS = {'primary_fits': 0, 'exclusion_control_fits': 0, 'internal_fits': 0, 'final_fits': 0, 'inner_fits': 0, 'refits': 0}
CONV = {'warnings': 0, 'max_n_iter': 0, 'missing_class_fits': 0}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- build
def step_features(step):
    """200 features from ONE step tensor (101,75,40): 10 time bins x 5x4 spatial blocks, mean pressure."""
    t = np.array_split(np.arange(101), 10); r = np.array_split(np.arange(75), 5); c = np.array_split(np.arange(40), 4)
    return np.array([step[np.ix_(ti, ri, ci)].mean() for ti in t for ri in r for ci in c])


def locate(pid):
    for s in SOURCES:
        d = s / 'py' / pid / 'BF' / 'W1'
        if (d / 'pipeline_2.npz').exists() and (d / 'metadata.csv').exists():
            return d
    return None


def eligible_steps(meta):
    m = meta[(meta.Side == 'Right') & (meta.Outlier == 0) & (meta.Exclude == 0) & (meta.Standing == 0) & (meta.Incomplete == 0)].sort_values('FootstepID')
    return m.index.to_numpy(), m.FootstepID.to_numpy()


def build():
    out = TASKDIR / 'build'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    pm = pd.read_csv(META); pm['pid'] = pm.ParticipantID.astype(int).map(lambda i: f'{i:03d}')
    bmi = pm['Weight (Kg)'] / (pm['Height (cm)'] / 100) ** 2
    pm['sex'] = (pm.Sex == 'Male').astype(int); pm['bmi25'] = (bmi >= 25).astype(int)
    X = []; ids = []; audit = []; src_hashes = {}
    for _, r in pm.sort_values('pid').iterrows():
        d = locate(r.pid)
        if d is None:
            audit.append({'pid': r.pid, 'status': 'MISSING_FILES'}); continue
        meta = pd.read_csv(d / 'metadata.csv'); rows, fsid = eligible_steps(meta)
        arr = np.load(d / 'pipeline_2.npz')['arr_0']; assert arr.shape[0] == len(meta) and arr.shape[1:] == (101, 75, 40)
        src_hashes[f'py/{r.pid}/BF/W1/pipeline_2.npz'] = sha(d / 'pipeline_2.npz'); src_hashes[f'py/{r.pid}/BF/W1/metadata.csv'] = sha(d / 'metadata.csv')
        if len(rows) < KMAX:
            audit.append({'pid': r.pid, 'status': 'EXCLUDED_FEWER_THAN_20', 'eligible_right_steps': int(len(rows))}); continue
        sel = rows[:KMAX]; feats = np.stack([step_features(arr[i]) for i in sel]); assert np.isfinite(feats).all()
        X.append(feats); ids.append(int(r.pid))
        audit.append({'pid': r.pid, 'status': 'INCLUDED', 'eligible_right_steps': int(len(rows)), 'total_steps': int(len(meta)), 'source': str(d.relative_to(ROOT)),
                      'bag_row_indices': ','.join(map(str, sel)), 'bag_footstep_ids': ','.join(map(str, fsid[:KMAX])), 'sex': int(r.sex), 'bmi25': int(r.bmi25)})
    X = np.stack(X); ids = np.array(ids); lab = pm.set_index(pm.ParticipantID.astype(int))
    y1 = lab.loc[ids, 'sex'].to_numpy(); y2 = lab.loc[ids, 'bmi25'].to_numpy()
    np.savez(out / 'features_cache.npz', X=X, person_ids=ids, sex=y1, bmi25=y2)
    pd.DataFrame(audit).to_csv(out / 'eligibility_audit.csv', index=False)
    # splits per label, recorded before any fit
    for L, col in LABELS.items():
        y = y1 if col == 'sex' else y2; rows = []
        for rep in range(3):
            for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(ids, y)):
                rows += [{'repeat': rep, 'fold': f, 'role': 'train', 'person_id': int(ids[i])} for i in tr] + [{'repeat': rep, 'fold': f, 'role': 'test', 'person_id': int(ids[i])} for i in te]
        pd.DataFrame(rows).to_csv(out / f'outer_splits_{L}.csv', index=False)
    info = {'included_people': int(len(ids)), 'excluded': [a for a in audit if a['status'] != 'INCLUDED'], 'sex_counts': np.bincount(y1).tolist(), 'bmi25_counts': np.bincount(y2).tolist(),
            'feature_dim': 200, 'kmax': KMAX, 'source_file_sha256': src_hashes, 'metadata_sha256': sha(META), 'checksum_file_sha256': sha(CHK),
            'download_log_sha256': sha(TASKDIR / 'download_log.json') if (TASKDIR / 'download_log.json').exists() else None,
            'features_cache_sha256': sha(out / 'features_cache.npz'), 'script_sha256': sha(Path(__file__)), 'addendum_sha256': sha(ADDENDUM)}
    (out / 'build_info.json').write_text(json.dumps(info, indent=2)); print(json.dumps({k: v for k, v in info.items() if k != 'source_file_sha256'}, indent=2))


# ----------------------------------------------------------------------------- task
class Task:
    def __init__(self, L):
        self.L = L; c = np.load(TASKDIR / 'build/features_cache.npz'); self.X = c['X']; self.ids = c['person_ids']; self.y = c[LABELS[L]].astype(int)
        self.A = None; self.nc = 2; self.pos = {int(i): j for j, i in enumerate(self.ids)}
        self.splits = pd.read_csv(TASKDIR / f'build/outer_splits_{L}.csv'); self.cfg = {'id_col': 'person_id', 'warm': WARM, 'batch': BATCH}

    def person_vec(self, X_src, p, trials):
        return np.median(X_src[p, trials], axis=0)

    def feats(self, X_src, acquired):
        f = np.stack([self.person_vec(X_src, p, t) for p, t in acquired.items()]); assert np.isfinite(f).all(); return f

    def test_feats(self, te):
        return np.stack([self.person_vec(self.X, p, np.arange(KMAX)) for p in te])

    def fit(self, feats, labels, seed, count_key, scaler_feats=None):
        s1 = StandardScaler().fit(feats); z = s1.transform(feats)
        pca = PCA(PCA_K, svd_solver='full', random_state=seed).fit(z); z = pca.transform(z); s2 = StandardScaler().fit(z); z = s2.transform(z)
        model = LogisticRegression(C=C_FIXED, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning); model.fit(z, labels)
            CONV['warnings'] += any(issubclass(w.category, ConvergenceWarning) for w in caught)
        CONV['max_n_iter'] = max(CONV['max_n_iter'], int(np.max(model.n_iter_))); FITS[count_key] += 1
        CONV['missing_class_fits'] += int(len(model.classes_) < self.nc)
        return (s1, pca, s2, model)

    def predict(self, pipe, feats):
        s1, pca, s2, model = pipe; p = model.predict_proba(s2.transform(pca.transform(s1.transform(feats))))
        if len(model.classes_) < self.nc:
            full = np.zeros((len(feats), self.nc)); full[:, model.classes_] = p; return full
        return p


def save_model(path, pipe, extra=None):
    s1, pca, s2, model = pipe
    np.savez(path, s1_mean=s1.mean_, s1_scale=s1.scale_, pca_mean=pca.mean_, pca_components=pca.components_, s2_mean=s2.mean_, s2_scale=s2.scale_,
             coef=model.coef_, intercept=model.intercept_, classes=model.classes_, n_iter=np.array([int(np.max(model.n_iter_))]), **(extra or {}))


def alloc_for(n, counts):
    p = counts / counts.sum(); raw = p * n; base = np.floor(raw).astype(int); rem = n - base.sum(); order = np.argsort(-(raw - base)); base[order[:rem]] += 1; return tuple(int(v) for v in base)


def freeze(stage):
    fp = STUDY / 'frozen_protocol_A5.json'
    if stage == 'pilot':
        bi = json.loads((TASKDIR / 'build/build_info.json').read_text())
        fr = {'status': 'FROZEN_A5_STEPUP', 'date': datetime.now().isoformat(timespec='seconds'), 'addendum_A5_sha256': sha(ADDENDUM), 'script_sha256': sha(Path(__file__)),
              'features_cache_sha256': bi['features_cache_sha256'], 'build_info_sha256': sha(TASKDIR / 'build/build_info.json'), 'grid': {'n': NS, 'k': KS, 'draws': DRAWS},
              'contrasts': [c[0] for c in CONTRASTS], 'C': C_FIXED, 'pca': PCA_K, 'kmax': KMAX, 'policies': POLICIES, 'policy_budgets': POLICY_BUDGETS, 'warm': WARM, 'batch': BATCH,
              'policy_seeds': POLICY_SEEDS, 'seed': SEED, 'bootstrap': BOOT, 'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
        fp.write_text(json.dumps(fr, indent=2))
    fr = json.loads(fp.read_text())
    assert fr['script_sha256'] == sha(Path(__file__)) and fr['addendum_A5_sha256'] == sha(ADDENDUM), 'changed after freeze'
    assert fr['features_cache_sha256'] == sha(TASKDIR / 'build/features_cache.npz'), 'feature cache changed'
    return fr


# ----------------------------------------------------------------------------- surface
def run_surface(stage, L):
    import restricted_acquisition_gaitrec as G
    out = TASKDIR / (f'surface_{L}_v1' if stage == 'surface' else 'pilot_v1')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(); (out / 'models').mkdir(); fr = freeze(stage); (out / 'frozen_protocol_A5_copy.json').write_text(json.dumps(fr, indent=2))
    task = Task(L); X = task.X; y = task.y; alloc_by_fold = {}
    cells = [(120, 5), (60, 10)] if stage == 'pilot' else [(n, k) for n in NS for k in KS]
    repeats = [0] if stage == 'pilot' else [0, 1, 2]; draws = 5 if stage == 'pilot' else DRAWS
    splits = task.splits[task.splits.repeat.isin(repeats)]
    mem = []; pred = []; audit = []; conv = []; t0 = time.perf_counter()
    for (repeat, fold), block in splits.groupby(['repeat', 'fold']):
        tr = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'train', 'person_id'])); te = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'test', 'person_id']))
        assert not set(tr) & set(te); tf = task.test_feats(te)
        # A5.1: allocation by largest remainder on THIS fold's training class counts (n=120 is then the whole training set); always feasible and nested.
        mins = np.bincount(y[tr], minlength=2); alloc = {n: alloc_for(n, mins) for n in NS}; alloc_by_fold[f'{repeat}_{fold}'] = {str(n): list(v) for n, v in alloc.items()}
        for n in NS:
            assert all(alloc[n][c] <= mins[c] for c in range(2)) and sum(alloc[n]) == n
        print(f'{stage} {L} r{repeat} f{fold}', flush=True)
        for draw in range(draws):
            rng = np.random.default_rng(SEED + repeat * 10000 + fold * 100 + draw)
            order = {c: rng.permutation(tr[y[tr] == c]) for c in range(2)}; trial_order = {int(p): rng.permutation(KMAX) for p in tr}
            for n in NS:
                people = np.sort(np.concatenate([order[c][:alloc[n][c]] for c in range(2)]))
                for k in KS:
                    if (n, k) not in cells:
                        continue
                    trials = {int(p): np.sort(trial_order[int(p)][:k]) for p in people}
                    assert tuple(np.bincount(y[people], minlength=2)) == alloc[n] and len(set(y[people])) == 2
                    X_nan = np.full_like(X, np.nan)
                    for p, ts in trials.items():
                        X_nan[p, ts] = X[p, ts]
                    X_noise = X_nan.copy(); m = np.isnan(X_noise); X_noise[m] = rng.normal(0, 1e3, m.sum()); labels = y[people]
                    t1 = time.perf_counter(); pipe = task.fit(task.feats(X_nan, {p: trials[p] for p in people}), labels, SEED + fold, 'primary_fits'); t2 = time.perf_counter()
                    pipe2 = task.fit(task.feats(X_noise, {p: trials[p] for p in people}), labels, SEED + fold, 'exclusion_control_fits')
                    p = task.predict(pipe, tf); p2 = task.predict(pipe2, tf)
                    ident = bool(np.array_equal(pipe[3].coef_, pipe2[3].coef_) and np.array_equal(p, p2)); assert ident
                    save_model(out / 'models' / f'r{repeat}_f{fold}_d{draw}_n{n}_k{k}.npz', pipe)
                    conv.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'n_iter': int(np.max(pipe[3].n_iter_)), 'reached_max_iter': int(np.max(pipe[3].n_iter_)) >= MAX_ITER, 'fit_seconds': t2 - t1})
                    mem.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'role': 'train', 'person_index': int(pp), 'person_id': int(task.ids[pp]),
                                'retained_trials': ','.join(str(t + 1) for t in trials[int(pp)]), 'class': int(y[pp])} for pp in people)
                    pred.extend({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'person_index': int(pp), 'person_id': int(task.ids[pp]), 'label': int(y[pp]),
                                 'p0': p[j, 0], 'p1': p[j, 1]} for j, pp in enumerate(te))
                    audit.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'recordings': n * k, 'class_counts': '/'.join(map(str, np.bincount(y[people], minlength=2))), 'exclusion_proof_identical': ident})
        mem.extend({'repeat': repeat, 'fold': fold, 'draw': -1, 'n': 0, 'k': 0, 'role': 'test', 'person_index': int(pp), 'person_id': int(task.ids[pp]), 'retained_trials': 'all20', 'class': int(y[pp])} for pp in te)
    pd.DataFrame(mem).to_csv(out / 'acquisition_memberships.csv', index=False); pred = pd.DataFrame(pred); pred.to_csv(out / 'heldout_predictions.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'acquisition_audit.csv', index=False); cv = pd.DataFrame(conv); cv.to_csv(out / 'convergence_audit.csv', index=False)
    # summarize with the generic (nc-aware) routine
    surface, con, fin, cellrows = summarize_surface(pred, y, cells, CONTRASTS if stage == 'surface' else [CONTRASTS[0]], draws)
    surface.to_csv(out / 'learning_surface.csv', index=False); con.to_csv(out / 'prespecified_contrasts.csv', index=False); fin.to_csv(out / 'finite_differences.csv', index=False)
    cellrows.to_csv(out / 'cell_metrics_by_repeat_draw.csv', index=False)
    (out / 'progress.json').write_text(json.dumps({**FITS, **CONV, 'stage': stage, 'label': L, 'wall_seconds': round(time.perf_counter() - t0, 1), 'allocation_by_fold': alloc_by_fold}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2)); print(surface.round(4).to_string(index=False)); print(con.round(4).to_string(index=False)); print(json.dumps(FITS))


def weighted_metrics(label, probs, W, nc):
    hard = probs.argmax(1); ba = np.zeros(W.shape[0])
    for c in range(nc):
        inc = (label == c).astype(float); cor = ((label == c) & (hard == c)).astype(float); ba += (W @ cor) / (W @ inc)
    ll = -(W @ np.log(np.clip(probs[np.arange(len(label)), label], 1e-15, 1))) / W.sum(1); return ba / nc, ll


def make_weights(y, boot, seed, nc):
    rng = np.random.default_rng(seed); W = np.ones((boot + 1, len(y))); idx = [np.where(y == c)[0] for c in range(nc)]
    for b in range(1, boot + 1):
        w = np.zeros(len(y))
        for c in range(nc):
            np.add.at(w, rng.choice(idx[c], len(idx[c]), replace=True), 1)
        W[b] = w
    return W


def summarize_surface(pred, y, cells, contrasts, draws):
    nc = 2; W = make_weights(y, BOOT, SEED + 999, nc); groups = {}
    for (r, d, n, k), g in pred.groupby(['repeat', 'draw', 'n', 'k']):
        g = g.sort_values('person_index'); assert g.person_index.tolist() == list(range(len(y)))
        groups[(r, d, n, k)] = weighted_metrics(g.label.to_numpy(), g[['p0', 'p1']].to_numpy(), W, nc)
    repeats = sorted(pred.repeat.unique()); keys = [(r, d) for r in repeats for d in range(draws)]
    def stats(stack):
        m = stack.mean(0); point = stack[:, 0].reshape(len(repeats), draws)
        return {'mean': m[0], 'ci_low': np.percentile(m[1:], 2.5), 'ci_high': np.percentile(m[1:], 97.5), 'within_repeat_sd_mean': point.std(axis=1, ddof=1).mean() if draws > 1 else float('nan'),
                'sd_of_repeat_means': point.mean(1).std(ddof=1) if len(repeats) > 1 else float('nan'), 'pairs_positive': int((stack[:, 0] > 0).sum())}
    ns = sorted({n for n, _ in cells}); ks = sorted({k for _, k in cells}); surface = []; con = []; fin = []
    for n, k in cells:
        ba = np.stack([groups[(r, d, n, k)][0] for r, d in keys]); ll = np.stack([groups[(r, d, n, k)][1] for r, d in keys]); sb = stats(ba); sl = stats(ll)
        surface.append({'n': n, 'k': k, 'recordings': n * k, 'balanced_accuracy': sb['mean'], 'ba_ci_low': sb['ci_low'], 'ba_ci_high': sb['ci_high'], 'ba_within_repeat_sd_mean': sb['within_repeat_sd_mean'],
                        'ba_sd_of_repeat_means': sb['sd_of_repeat_means'], 'log_loss': sl['mean'], 'll_ci_low': sl['ci_low'], 'll_ci_high': sl['ci_high'], 'n_values_averaged': len(keys)})
    for name, (na, ka), (nb, kb) in contrasts:
        if (na, ka) in cells and (nb, kb) in cells:
            for metric, j in [('balanced_accuracy', 0), ('log_loss', 1)]:
                st = np.stack([groups[(r, d, na, ka)][j] - groups[(r, d, nb, kb)][j] for r, d in keys])
                con.append({'contrast': name, 'metric': metric, 'recordings_A': na * ka, 'recordings_B': nb * kb, 'paired_pairs': len(keys), **stats(st)})
    for n, k in cells:
        for n2, k2 in cells:
            if (n2 == n and ks.index(k2) == ks.index(k) + 1) or (k2 == k and ns.index(n2) == ns.index(n) + 1):
                st = np.stack([groups[(r, d, n2, k2)][0] - groups[(r, d, n, k)][0] for r, d in keys]); s = stats(st)
                fin.append({'from': f'{n}x{k}', 'to': f'{n2}x{k2}', 'axis': 'k' if n2 == n else 'n', 'mean_difference': s['mean'], 'ci_low': s['ci_low'], 'ci_high': s['ci_high']})
    cellrows = [{'repeat': r, 'draw': d, 'n': n, 'k': k, 'balanced_accuracy': v[0][0], 'log_loss': v[1][0]} for (r, d, n, k), v in groups.items()]
    return pd.DataFrame(surface), pd.DataFrame(con), pd.DataFrame(fin), pd.DataFrame(cellrows)


# ----------------------------------------------------------------------------- policy (A4 rules with KMAX)
def ba_present(label, hard, nc):
    present = [c for c in range(nc) if (label == c).any()]; return float(np.mean([(hard[label == c] == c).mean() for c in present]))


def internal_gains(task, acquired, labels_of, rng, nc):
    people = np.array(list(acquired)); labels = np.array([labels_of[p] for p in people]); reps = []
    warnings.filterwarnings('ignore', category=RuntimeWarning); warnings.filterwarnings('ignore', category=UserWarning)
    for rep in range(3):
        skf = StratifiedKFold(3, shuffle=True, random_state=int(rng.integers(1 << 31))); res = {'full': [], 'people': [], 'trial': []}; removed = {'people': [], 'trial': []}
        for it, iv in skf.split(people, labels):
            val = {int(people[j]): acquired[int(people[j])] for j in iv}; vf = task.feats(task.X, val); vl = labels[iv]
            tr_full = {int(people[j]): acquired[int(people[j])] for j in it}; keep = rng.permutation(it)[:max(nc, int(round(0.8 * len(it))))]
            tr_people = {int(people[j]): acquired[int(people[j])] for j in keep}; tr_trial = {}
            for j in it:
                p = int(people[j]); t = acquired[p]; tr_trial[p] = np.sort(np.delete(t, rng.integers(len(t)))) if len(t) >= 2 else t
            for key, tr in {'full': tr_full, 'people': tr_people, 'trial': tr_trial}.items():
                tl = np.array([labels_of[p] for p in tr])
                if len(set(tl)) < nc or len(tl) <= PCA_K:
                    res[key].append(np.nan); continue
                pipe = task.fit(task.feats(task.X, tr), tl, SEED + rep, 'internal_fits'); res[key].append(ba_present(vl, task.predict(pipe, vf).argmax(1), nc))
            removed['people'].append(sum(len(t) for t in tr_full.values()) - sum(len(t) for t in tr_people.values())); removed['trial'].append(sum(len(t) for t in tr_full.values()) - sum(len(t) for t in tr_trial.values()))
        full = np.nanmean(res['full']) if np.isfinite(res['full']).any() else np.nan
        gp = (full - np.nanmean(res['people'])) / max(1, np.mean(removed['people'])) if np.isfinite(res['people']).any() else np.nan
        gt = (full - np.nanmean(res['trial'])) / max(1, np.mean(removed['trial'])) if np.isfinite(res['trial']).any() else np.nan
        reps.append({'replicate': rep, 'ba_full': full, 'ba_people_reduced': np.nanmean(res['people']) if np.isfinite(res['people']).any() else np.nan,
                     'ba_trial_reduced': np.nanmean(res['trial']) if np.isfinite(res['trial']).any() else np.nan, 'removed_people_recordings': float(np.mean(removed['people'])),
                     'removed_trial_recordings': float(np.mean(removed['trial'])), 'gain_people': gp, 'gain_trial': gt})
    return reps


def run_policy(task, policy, pool_order, trial_order, labels_of, B, rng, log):
    acquired = {}; recruited = []; n_rec = 0; nc = task.nc
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
        nonlocal n_rec; added = 0
        for p in sorted(recruited, key=lambda q: (len(acquired[q]), recruited.index(q))):
            if added >= count or n_rec >= B:
                break
            if len(acquired[p]) < KMAX:
                acquired[p] = np.array(trial_order[p][:len(acquired[p]) + 1]); n_rec += 1; added += 1
        return added
    if policy.startswith('FIX') or policy == 'CONV5':
        k = 5 if policy == 'CONV5' else int(policy[3:])
        while n_rec < B and recruit(k):
            pass
        while n_rec < B:
            if add_trials(B - n_rec) == 0:
                raise RuntimeError('budget exceeds pool capacity')
    else:
        for _ in range(WARM):
            recruit(2)
        step = 0
        while n_rec < B:
            b = min(BATCH, B - n_rec)
            if policy == 'RANDOM':
                action = 'people' if rng.random() < 0.5 else 'trials'
            else:
                reps = internal_gains(task, acquired, labels_of, rng, nc); chosen = int(rng.integers(3)); est = reps[chosen]
                undefined = bool(np.isnan(est['gain_people']) or np.isnan(est['gain_trial']))
                action = 'people' if undefined else ('people' if est['gain_people'] >= est['gain_trial'] else 'trials')
                for r in reps:
                    log.append({'step': step, 'n_people': len(acquired), 'recordings': n_rec, **r, 'chosen_replicate': chosen, 'action': action, 'undefined_estimate': undefined})
            if action == 'people':
                got = 0
                while got < b and n_rec < B and recruit(1):
                    got += 1
                if got < b and n_rec < B:
                    add_trials(b - got)
            else:
                added = add_trials(b)
                while added < b and n_rec < B and recruit(1):
                    added += 1
            if policy == 'RANDOM':
                log.append({'step': step, 'n_people': len(acquired), 'recordings': n_rec, 'action': action})
            step += 1
    assert sum(len(t) for t in acquired.values()) == B and all(len(set(t)) == len(t) <= KMAX for t in acquired.values())
    return acquired


def summarize_policy(pred, y_all, nc, policies, budgets):
    W = make_weights(y_all, BOOT, SEED + 999, nc); metrics = {}
    for (r, s, B, pol), g in pred.groupby(['repeat', 'seed', 'budget', 'policy']):
        g = g.sort_values('person_index'); assert g.person_index.tolist() == list(range(len(y_all)))
        metrics[(r, s, B, pol)] = weighted_metrics(g.label.to_numpy(), g[['p0', 'p1']].to_numpy(), W, nc)
    repeats = sorted(pred.repeat.unique()); seeds = sorted(pred.seed.unique())
    def stats(stack):
        m = stack.mean(0); point = stack[:, 0].reshape(len(repeats), len(seeds))
        return {'mean': m[0], 'ci_low': np.percentile(m[1:], 2.5), 'ci_high': np.percentile(m[1:], 97.5), 'within_repeat_sd_mean': point.std(axis=1, ddof=1).mean() if len(seeds) > 1 else float('nan'),
                'sd_of_repeat_means': point.mean(1).std(ddof=1) if len(repeats) > 1 else float('nan'), 'pairs_positive': int((stack[:, 0] > 0).sum()), 'pairs': stack.shape[0]}
    out = []
    for B in budgets:
        for pol in policies:
            for mi, mname in enumerate(['balanced_accuracy', 'log_loss']):
                st = np.stack([metrics[(r, s, B, pol)][mi] for r in repeats for s in seeds]); sb = stats(st); out.append({'budget': B, 'policy': pol, 'metric': mname, **{k: v for k, v in sb.items() if k != 'pairs_positive'}})
        fixed = [p for p in policies if p.startswith('FIX') or p == 'CONV5']; oracle = max(fixed, key=lambda p: np.mean([metrics[(r, s, B, p)][0][0] for r in repeats for s in seeds]))
        for ref in ['CONV5', 'RANDOM', oracle]:
            for mi, mname in enumerate(['balanced_accuracy', 'log_loss']):
                st = np.stack([metrics[(r, s, B, 'MARGIN')][mi] - metrics[(r, s, B, ref)][mi] for r in repeats for s in seeds])
                out.append({'budget': B, 'policy': f'MARGIN_minus_{ref}' + ('_ORACLEFIX' if ref == oracle and ref not in ('CONV5', 'RANDOM') else ''), 'metric': mname, **stats(st)})
    return pd.DataFrame(out)


def run_policy_stage(L):
    out = TASKDIR / f'policy_{L}_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(); (out / 'models').mkdir(); fr = freeze('policy'); (out / 'frozen_protocol_A5_copy.json').write_text(json.dumps(fr, indent=2))
    task = Task(L); labels_of = {int(i): int(c) for i, c in enumerate(task.y)}; acq_rows = []; pred_rows = []; dec_rows = []; t0 = time.perf_counter()
    for (repeat, fold), block in task.splits.groupby(['repeat', 'fold']):
        tr = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'train', 'person_id'])); te = np.array(sorted(task.pos[int(p)] for p in block.loc[block.role == 'test', 'person_id']))
        tf = task.test_feats(te); tl = task.y[te]
        for seed in range(POLICY_SEEDS):
            print(f'policy {L} r{repeat} f{fold} seed{seed}', flush=True)
            crn = np.random.default_rng(SEED + repeat * 100000 + fold * 1000 + seed); pool_order = [int(p) for p in crn.permutation(tr)]; trial_order = {int(p): crn.permutation(KMAX) for p in tr}
            for B in POLICY_BUDGETS:
                for pol in POLICIES:
                    prng = np.random.default_rng(SEED + repeat * 100000 + fold * 1000 + seed * 20 + POLICY_BUDGETS.index(B) * 10 + POLICIES.index(pol))
                    log = []; acquired = run_policy(task, pol, pool_order, trial_order, labels_of, B, prng, log)
                    dec_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, **row} for row in log]
                    labels = np.array([labels_of[p] for p in acquired]); X_nan = np.full_like(task.X, np.nan)
                    for p, t in acquired.items():
                        X_nan[p, t] = task.X[p, t]
                    X_noise = X_nan.copy(); m = np.isnan(X_noise); X_noise[m] = prng.normal(0, 1e3, m.sum())
                    pipe = task.fit(task.feats(X_nan, acquired), labels, SEED + fold, 'final_fits'); pipe2 = task.fit(task.feats(X_noise, acquired), labels, SEED + fold, 'exclusion_control_fits')
                    p = task.predict(pipe, tf); p2 = task.predict(pipe2, tf); assert np.array_equal(pipe[3].coef_, pipe2[3].coef_) and np.array_equal(p, p2)
                    save_model(out / 'models' / f'r{repeat}_f{fold}_s{seed}_B{B}_{pol}.npz', pipe); missing = int(len(pipe[3].classes_) < task.nc)
                    acq_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'person_id': int(task.ids[pp]), 'trials': ','.join(str(t + 1) for t in np.sort(tt)),
                                  'n_trials': len(tt), 'class': labels_of[pp], 'model_missing_class': missing} for pp, tt in acquired.items()]
                    pred_rows += [{'repeat': repeat, 'fold': fold, 'seed': seed, 'budget': B, 'policy': pol, 'person_index': int(pp), 'label': int(tl[j]), 'p0': p[j, 0], 'p1': p[j, 1]} for j, pp in enumerate(te)]
    acq = pd.DataFrame(acq_rows); acq.to_csv(out / 'acquisitions.csv', index=False); pred = pd.DataFrame(pred_rows); pred.to_csv(out / 'heldout_predictions.csv', index=False)
    pd.DataFrame(dec_rows).to_csv(out / 'policy_decisions.csv', index=False)
    summ = summarize_policy(pred, task.y, task.nc, POLICIES, POLICY_BUDGETS); summ.to_csv(out / 'policy_summary.csv', index=False)
    comp = acq.groupby(['repeat', 'fold', 'seed', 'budget', 'policy']).agg(people=('person_index', 'nunique'), trials_per_person=('n_trials', 'mean')).reset_index() \
        .groupby(['budget', 'policy']).agg(mean_people=('people', 'mean'), sd_people=('people', 'std'), mean_trials_per_person=('trials_per_person', 'mean')).reset_index()
    comp.to_csv(out / 'acquired_composition.csv', index=False)
    (out / 'progress.json').write_text(json.dumps({**FITS, **CONV, 'stage': 'policy', 'label': L, 'wall_seconds': round(time.perf_counter() - t0, 1)}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2)); print(comp.round(2).to_string(index=False)); print(summ[summ.metric == 'balanced_accuracy'].round(4).to_string(index=False)); print(json.dumps(FITS))


# ----------------------------------------------------------------------------- sensitivity (A2 rule) on primary cells, L1
def run_sensitivity(L):
    out = TASKDIR / f'sensitivity_{L}_v1'; src = TASKDIR / f'surface_{L}_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(); (out / 'models').mkdir(); fr = freeze('sensitivity'); (out / 'frozen_protocol_A5_copy.json').write_text(json.dumps(fr, indent=2))
    task = Task(L); mem = pd.read_csv(src / 'acquisition_memberships.csv'); gp = pd.read_csv(src / 'heldout_predictions.csv'); CS = [0.1, 1.0, 10.0]; cells = [(120, 5), (60, 10)]
    inner_rows = []; sel_rows = []; preds = []; t0 = time.perf_counter()
    for (repeat, fold), tb in mem[mem.role == 'test'].groupby(['repeat', 'fold']):
        te = np.sort(tb.person_index.to_numpy()); tf = task.test_feats(te)
        for draw in range(DRAWS):
            for n, k in cells:
                cell = mem[(mem.role == 'train') & (mem.repeat == repeat) & (mem.fold == fold) & (mem.draw == draw) & (mem.n == n) & (mem.k == k)]
                people = cell.person_index.to_numpy(); trials = {int(p): np.array([int(t) - 1 for t in ts.split(',')]) for p, ts in zip(cell.person_index, cell.retained_trials)}
                feats = task.feats(task.X, {int(p): trials[int(p)] for p in people}); labels = task.y[people]
                skf = StratifiedKFold(3, shuffle=True, random_state=SEED + 7 + repeat * 10000 + fold * 100 + draw); losses = {C: [] for C in CS}
                for i, (it, iv) in enumerate(skf.split(people, labels)):
                    for C in CS:
                        s1 = StandardScaler().fit(feats[it]); z = s1.transform(feats[it]); pca = PCA(PCA_K, svd_solver='full', random_state=SEED).fit(z); z = pca.transform(z); s2 = StandardScaler().fit(z)
                        mdl = LogisticRegression(C=C, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=SEED).fit(s2.transform(z), labels[it]); FITS['inner_fits'] += 1
                        pv = mdl.predict_proba(s2.transform(pca.transform(s1.transform(feats[iv])))); ll = float(-np.mean(np.log(np.clip(pv[np.arange(len(iv)), labels[iv]], 1e-15, 1)))); losses[C].append(ll)
                        inner_rows.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'inner_fold': i, 'C': C, 'inner_val_log_loss': ll})
                means = {C: float(np.mean(losses[C])) for C in CS}; best = min(CS, key=lambda C: (means[C], C))
                sel_rows.append({'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, **{f'mean_inner_ll_C{C}': means[C] for C in CS}, 'selected_C': best})
                s1 = StandardScaler().fit(feats); z = s1.transform(feats); pca = PCA(PCA_K, svd_solver='full', random_state=SEED + fold).fit(z); z = pca.transform(z); s2 = StandardScaler().fit(z)
                mdl = LogisticRegression(C=best, class_weight='balanced', max_iter=MAX_ITER, tol=1e-7, random_state=SEED + fold).fit(s2.transform(z), labels); FITS['refits'] += 1
                p = mdl.predict_proba(s2.transform(pca.transform(s1.transform(tf)))); save_model(out / 'models' / f'r{repeat}_f{fold}_d{draw}_n{n}_k{k}.npz', (s1, pca, s2, mdl), {'C': np.array([best])})
                preds += [{'repeat': repeat, 'fold': fold, 'draw': draw, 'n': n, 'k': k, 'person_index': int(pp), 'label': int(task.y[pp]), 'p0': p[j, 0], 'p1': p[j, 1], 'selected_C': best} for j, pp in enumerate(te)]
    pd.DataFrame(inner_rows).to_csv(out / 'inner_tuning_losses.csv', index=False); sel = pd.DataFrame(sel_rows); sel.to_csv(out / 'selected_C.csv', index=False)
    pred = pd.DataFrame(preds); pred.to_csv(out / 'heldout_predictions_tuned.csv', index=False)
    tuned_s, tuned_c, _, _ = summarize_surface(pred, task.y, cells, [CONTRASTS[0]], DRAWS); fixed_s, fixed_c, _, _ = summarize_surface(gp[gp.apply(lambda r: (r.n, r.k) in cells, axis=1)], task.y, cells, [CONTRASTS[0]], DRAWS)
    tuned_s['regime'] = 'tuned'; fixed_s['regime'] = 'fixed'; tuned_c['regime'] = 'tuned'; fixed_c['regime'] = 'fixed'
    pd.concat([tuned_s, fixed_s]).to_csv(out / 'cells_tuned_vs_fixed.csv', index=False); pd.concat([tuned_c, fixed_c]).to_csv(out / 'contrast_tuned_vs_fixed.csv', index=False)
    sel.groupby(['n', 'k']).selected_C.value_counts().unstack(fill_value=0).reset_index().to_csv(out / 'selected_C_distribution.csv', index=False)
    (out / 'progress.json').write_text(json.dumps({**FITS, 'stage': 'sensitivity', 'label': L, 'wall_seconds': round(time.perf_counter() - t0, 1)}, indent=2))
    manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out / 'output_manifest.json').write_text(json.dumps(manifest, indent=2)); print(pd.concat([tuned_c, fixed_c]).round(4).to_string(index=False)); print(json.dumps(FITS))


if __name__ == '__main__':
    stage = sys.argv[1]; L = sys.argv[2] if len(sys.argv) > 2 else 'L1'
    try:
        {'build': lambda: build(), 'pilot': lambda: run_surface('pilot', 'L1'), 'surface': lambda: run_surface('surface', L), 'policy': lambda: run_policy_stage(L), 'sensitivity': lambda: run_sensitivity(L)}[stage]()
    except Exception as exc:
        name = {'build': 'build', 'pilot': 'pilot_v1', 'surface': f'surface_{L}_v1', 'policy': f'policy_{L}_v1', 'sensitivity': f'sensitivity_{L}_v1'}[stage]; out = TASKDIR / name
        if out.exists():
            failed = TASKDIR / f'{name}_FAILED_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; out.rename(failed)
            (failed / 'FAILURE.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(), 'fits_before_failure': FITS}, indent=2))
        raise
