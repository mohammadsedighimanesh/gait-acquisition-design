"""Nested person-level test of learned set pooling and coherence-sensitive aggregation."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'idea1_2026/learned_multitrial_aggregation_pilot_v1'
RAW = ROOT / 'idea1_2026/corrected_raw_pipeline_diagnostic_v2/raw_trial_level_features_v2.csv'
SPLITS = ROOT / 'idea1_2026/original_paper_incremental_value_pilot_v1/outer_splits_before_fit.csv'
SEED = 20261015
CS = [.1, 1., 10.]
METHODS = ['Mean_LR', 'Median_LR', 'Mean_SD_LR', 'Trial_vote',
           'DeepSets_mean', 'Attention_MIL', 'Coherence_attention', 'Coherence_shuffled']
SCENARIOS = ['Clean', 'One_mislinked_trial', 'Two_mislinked_trials', 'Drop_one_trial']
COUNTS = {'logistic_fits': 0, 'neural_optimizer_starts': 0}
OPTIMIZATION = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load():
    df = pd.read_csv(RAW).sort_values(['participant_id', 'trial_id']).reset_index(drop=True)
    assert len(df) == 350 and df.groupby('participant_id').size().eq(5).all()
    angles = [f'R_{j}_{p}' for j in ['ANK', 'MT', 'MP'] for p in ['AB', 'INV', 'DF']]
    blocks = []; names = []
    for c in angles:
        blocks.extend([df[f'{c}_mean'].to_numpy(), (df[f'{c}_max'] - df[f'{c}_min']).to_numpy()])
        names.extend([f'{c}_mean', f'{c}_range'])
    pressure = ['PMAX_mean', 'pressure_positive_mean', 'pressure_positive_std', 'pressure_positive_max',
                'pressure_active_fraction', 'pressure_total_mean', 'pressure_total_std', 'pressure_total_max',
                'pressure_peak_frame', 'pressure_temporal_center']
    names += pressure; blocks += [df[c].to_numpy() for c in pressure]
    X = np.column_stack(blocks).reshape(70, 5, 28)
    Acols = [c for c in df if c.endswith('_first') and 'FINGER_SHAPE' not in c]
    A = df.groupby('participant_id')[Acols].first().sort_index().to_numpy()
    toe = df.groupby('participant_id')[['RF_FINGER_SHAPE_first', 'LF_FINGER_SHAPE_first']].first().sort_index()
    A = np.column_stack([A] + [(toe[c].to_numpy() == k).astype(float) for c in toe for k in [1, 2, 3]])
    y = df.groupby('participant_id').y.first().sort_index().to_numpy(dtype=int)
    assert A.shape == (70, 19) and np.isfinite(X).all() and np.isfinite(A).all()
    assert not any('FPI' in c.upper() or c.startswith('M_') for c in names + Acols)
    return X, A, y, names, Acols


class Prepared:
    def fit(self, X, A):
        self.xscale = StandardScaler().fit(X.reshape(-1, 28))
        self.ascale = StandardScaler().fit(A)
        return self

    def transform(self, X, A):
        return self.xscale.transform(X.reshape(-1, 28)).reshape(X.shape), self.ascale.transform(A)


def pooled(method, X, A):
    if method == 'Mean_LR':
        return np.column_stack([X.mean(axis=1), A])
    if method == 'Median_LR':
        return np.column_stack([np.median(X, axis=1), A])
    if method == 'Mean_SD_LR':
        return np.column_stack([X.mean(axis=1), X.std(axis=1, ddof=1), A])
    return np.column_stack([X.reshape(-1, 28), np.repeat(A, X.shape[1], axis=0)])


class SetModel(torch.nn.Module):
    def __init__(self, method):
        super().__init__(); self.method = method
        self.encoder = torch.nn.Linear(28, 8)
        self.readout = torch.nn.Linear(27, 3)
        if method != 'DeepSets_mean':
            self.attention = torch.nn.Parameter(torch.zeros(8))
        if method.startswith('Coherence'):
            self.coherence_raw = torch.nn.Parameter(torch.tensor(-3.1780538303479458))

    def forward(self, X, A):
        z = torch.tanh(self.encoder(X))
        if self.method == 'DeepSets_mean':
            weights = torch.full(X.shape[:2], 1 / X.shape[1], dtype=X.dtype)
        else:
            score = z @ self.attention
            if self.method.startswith('Coherence'):
                center = X.median(dim=1, keepdim=True).values
                distance = ((X - center) ** 2).mean(dim=2)
                if self.method == 'Coherence_shuffled':
                    distance = torch.roll(distance, 1, dims=1)
                gamma = 5 * torch.sigmoid(self.coherence_raw)
                score = score - gamma * distance
            weights = torch.softmax(score, dim=1)
        summary = (z * weights[:, :, None]).sum(dim=1)
        return self.readout(torch.cat([summary, A], dim=1)), weights


def train(method, X, A, y, c, seed, scope):
    if method in METHODS[:4]:
        features = pooled(method, X, A); scaler = StandardScaler().fit(features)
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=3000, tol=1e-7, random_state=seed)
        if method == 'Trial_vote':
            model.fit(scaler.transform(features), np.repeat(y, X.shape[1]), sample_weight=np.full(len(features), 1 / X.shape[1]))
        else:
            model.fit(scaler.transform(features), y)
        COUNTS['logistic_fits'] += 1
        return model, scaler
    tx = torch.from_numpy(X); ta = torch.from_numpy(A); ty = torch.from_numpy(y)
    sample_weights = torch.from_numpy(len(y) / (3 * np.bincount(y, minlength=3)[y]))
    candidates = []
    for start in range(2):
        torch.manual_seed(seed + start * 10000)
        model = SetModel(method)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, max_eval=160,
            tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn='strong_wolfe')
        def objective():
            logits, _ = model(tx, ta)
            ce = (sample_weights * torch.nn.functional.cross_entropy(logits, ty, reduction='none')).mean()
            penalty = sum((p ** 2).sum() for name, p in model.named_parameters()
                          if 'bias' not in name and name != 'coherence_raw') / (2 * c * len(y))
            return ce + penalty
        def closure():
            optimizer.zero_grad(); loss = objective(); loss.backward(); return loss
        optimizer.step(closure)
        optimizer.zero_grad(); loss = objective(); loss.backward()
        value = float(loss.detach()); gradient = max(float(p.grad.abs().max()) for p in model.parameters())
        assert np.isfinite(value) and np.isfinite(gradient)
        state = optimizer.state[next(model.parameters())]
        OPTIMIZATION.append({'scope': scope, 'method': method, 'C': c, 'start': start,
                             'training_objective': value, 'gradient_max': gradient,
                             'iterations': int(state.get('n_iter', 0)), 'function_evaluations': int(state.get('func_evals', 0))})
        COUNTS['neural_optimizer_starts'] += 1
        candidates.append((value, start, model))
    return min(candidates, key=lambda item: (item[0], item[1]))[2], None


def predict(method, trained, X, A):
    model, scaler = trained
    if method in METHODS[:4]:
        p = model.predict_proba(scaler.transform(pooled(method, X, A)))
        if method == 'Trial_vote':
            p = p.reshape(len(X), X.shape[1], 3).mean(axis=1)
        return p, np.full(X.shape[:2], 1 / X.shape[1])
    with torch.no_grad():
        logits, w = model(torch.from_numpy(X), torch.from_numpy(A))
        return torch.softmax(logits, dim=1).numpy(), w.numpy()


def save_model(path, method, trained, prep):
    model, scaler = trained
    arrays = {'x_mean': prep.xscale.mean_, 'x_scale': prep.xscale.scale_,
              'a_mean': prep.ascale.mean_, 'a_scale': prep.ascale.scale_}
    if method in METHODS[:4]:
        arrays.update(pool_mean=scaler.mean_, pool_scale=scaler.scale_, coef=model.coef_, intercept=model.intercept_)
    else:
        arrays.update({f'network__{name}': value.detach().numpy() for name, value in model.state_dict().items()})
    np.savez_compressed(path, **arrays)


def stress(X, tr, te, repeat, fold):
    output = {'Clean': X[te].copy(), 'One_mislinked_trial': X[te].copy(),
              'Two_mislinked_trials': X[te].copy()}; dropped = []; logs = []
    for j, pid in enumerate(te):
        rng = np.random.default_rng(SEED + 90000 + repeat * 1000 + fold * 100 + int(pid))
        slots = rng.choice(5, 2, replace=False); donors = rng.choice(tr, 2, replace=False); trials = rng.integers(0, 5, 2)
        output['One_mislinked_trial'][j, slots[0]] = X[donors[0], trials[0]]
        for slot, donor, tid in zip(slots, donors, trials):
            output['Two_mislinked_trials'][j, slot] = X[donor, tid]
            logs.append({'repeat': repeat, 'fold': fold, 'participant_id': int(pid + 1),
                         'replaced_slot': int(slot), 'donor_id': int(donor + 1), 'donor_trial': int(tid),
                         'order': int(np.flatnonzero(slots == slot)[0])})
        dropped.append(np.delete(X[pid], slots[0], axis=0))
    output['Drop_one_trial'] = np.stack(dropped)
    return output, logs


def selftest():
    rng = np.random.default_rng(3); X = rng.normal(size=(12, 5, 28)); A = rng.normal(size=(12, 19)); y = np.arange(12) % 3
    prep = Prepared().fit(X[:9], A[:9]); original = prep.xscale.mean_.copy()
    prep.transform(X[9:] + 100, A[9:] + 100); np.testing.assert_array_equal(original, prep.xscale.mean_)
    for method in ['DeepSets_mean', 'Attention_MIL', 'Coherence_attention']:
        torch.manual_seed(4); model = SetModel(method)
        a, w = predict(method, (model, None), X, A)
        b, _ = predict(method, (model, None), X[:, [4, 0, 3, 1, 2]], A)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(w.sum(axis=1), 1., atol=1e-12)
        loss = model(torch.from_numpy(X), torch.from_numpy(A))[0].square().mean(); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model = SetModel('Coherence_attention')
    tx = torch.from_numpy(X); ta = torch.from_numpy(A)
    loss = model(tx, ta)[0].square().mean(); loss.backward()
    analytic = float(model.coherence_raw.grad); original = float(model.coherence_raw.detach())
    with torch.no_grad():
        model.coherence_raw.fill_(original + 1e-5); hi = float(model(tx, ta)[0].square().mean())
        model.coherence_raw.fill_(original - 1e-5); lo = float(model(tx, ta)[0].square().mean())
    np.testing.assert_allclose(analytic, (hi - lo) / 2e-5, rtol=1e-4, atol=1e-8)
    print('SELFTEST PASS: train-only scaling, genuine-set invariance, normalized weights, finite autodiff and coherence gradient.')


def run():
    if OUT.exists():
        raise FileExistsError(f'Existing outputs preserved: {OUT}')
    OUT.mkdir(); (OUT / 'models').mkdir()
    X, A, y, names, acols = load()
    config = {'seed': SEED, 'C_grid': CS, 'methods': METHODS, 'scenarios': SCENARIOS,
        'trial_features': names, 'anthropometric_numeric_fields': acols, 'hidden_units': 8,
        'neural_initializations': 2, 'optimizer': 'CPU float64 PyTorch LBFGS, max_iter100, max_eval160, strong_wolfe',
        'model_selection': 'inner clean log loss; ties smaller C; two neural starts selected by training objective only',
        'primary_contrast': 'Coherence_attention versus Attention_MIL under one simulated mislinked trial, BA',
        'continuation_rule': 'primary conditional lower CI>0 and gain>=0.03; clean BA lower difference CI> -0.02; benefit beyond median pooling and shuffled coherence; no novelty/external certification',
        'stress_semantics': 'held-out recipient receives 1 or 2 real trials from training donors, label-blind selection; model trained on clean data only; not evidence of actual record corruption',
        'quality': 'mean squared standardized trial distance to within-bag coordinate median; learned gamma in [0,5]',
        'acquisition': 'five trials per person; person-level label; no moments or FPI score predictors',
        'torch': torch.__version__, 'python': sys.version, 'source_sha256': sha(RAW), 'splits_sha256': sha(SPLITS),
        'script_sha256': sha(Path(__file__)), 'exploratory': True, 'previously_examined_cohort': True}
    dump('frozen_config.json', config)
    np.savez_compressed(OUT / 'source_feature_cache.npz', X=X, A=A, labels=y)
    splits = pd.read_csv(SPLITS); splits.to_csv(OUT / 'outer_splits_before_fit.csv', index=False)
    predictions = []; weights = []; audit = []; tuning = []; donors = []; selected = []
    for (repeat, fold), group in splits.groupby(['repeat', 'fold']):
        tr = np.sort(group.loc[group.role == 'train', 'participant_id'].to_numpy()) - 1
        te = np.sort(group.loc[group.role == 'test', 'participant_id'].to_numpy()) - 1
        seed = SEED + int(repeat) * 1000 + int(fold) * 100
        print(f'Repeat {repeat + 1}/3 fold {fold + 1}/5', flush=True)
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed + 50).split(tr, y[tr])); batches = []
        for k, (it, iv) in enumerate(inner):
            prep = Prepared().fit(X[tr[it]], A[tr[it]])
            batches.append((prep.transform(X[tr[it]], A[tr[it]]), prep.transform(X[tr[iv]], A[tr[iv]])))
            audit.extend({'repeat': repeat, 'fold': fold, 'inner_fold': k, 'role': role, 'participant_id': int(pid + 1)}
                         for role, ids in [('train', tr[it]), ('validation', tr[iv])] for pid in ids)
        prep = Prepared().fit(X[tr], A[tr]); xt, at = prep.transform(X[tr], A[tr])
        conditions, logs = stress(X, tr, te, int(repeat), int(fold)); donors += logs
        for method in METHODS:
            scores = []
            for c in CS:
                loss = []
                for k, ((it, iv), ((ix, ia), (vx, va))) in enumerate(zip(inner, batches)):
                    trained = train(method, ix, ia, y[tr[it]], c, seed + k, f'r{repeat}_f{fold}_inner{k}')
                    p, _ = predict(method, trained, vx, va); loss.append(log_loss(y[tr[iv]], p, labels=[0, 1, 2]))
                scores.append(float(np.mean(loss)))
            best = min(range(3), key=lambda i: (scores[i], CS[i])); c = CS[best]
            tuning.extend({'repeat': repeat, 'fold': fold, 'method': method, 'C': value,
                           'inner_log_loss': scores[i], 'selected': i == best} for i, value in enumerate(CS))
            trained = train(method, xt, at, y[tr], c, seed, f'r{repeat}_f{fold}_outer')
            save_model(OUT / 'models' / f'r{repeat}_f{fold}_{method}.npz', method, trained, prep)
            selected.append({'repeat': repeat, 'fold': fold, 'method': method, 'C': c})
            for scenario, raw_x in conditions.items():
                vx, va = prep.transform(raw_x, A[te]); p, w = predict(method, trained, vx, va)
                assert np.isfinite(p).all()
                for j, pid in enumerate(te):
                    predictions.append({'repeat': repeat, 'fold': fold, 'method': method, 'scenario': scenario,
                        'participant_id': int(pid + 1), 'y_true': int(y[pid]), 'y_pred': int(p[j].argmax()),
                        **dict(zip(['p0', 'p1', 'p2'], p[j]))})
                    weights.extend({'repeat': repeat, 'fold': fold, 'method': method, 'scenario': scenario,
                        'participant_id': int(pid + 1), 'position': slot, 'weight': float(value)} for slot, value in enumerate(w[j]))
        pd.DataFrame(predictions).to_csv(OUT / 'predictions_checkpoint.csv', index=False)
        pd.DataFrame(OPTIMIZATION).to_csv(OUT / 'optimizer_checkpoint.csv', index=False)
        dump('progress.json', {'last_completed': [int(repeat), int(fold)], **COUNTS})
    pred = pd.DataFrame(predictions); pred.to_csv(OUT / 'heldout_predictions.csv', index=False)
    pd.DataFrame(weights).to_csv(OUT / 'trial_weights.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'inner_split_audit.csv', index=False)
    pd.DataFrame(tuning).to_csv(OUT / 'inner_tuning.csv', index=False)
    pd.DataFrame(donors).to_csv(OUT / 'stress_donor_audit.csv', index=False)
    pd.DataFrame(selected).to_csv(OUT / 'selected_models.csv', index=False)
    pd.DataFrame(OPTIMIZATION).to_csv(OUT / 'optimizer_diagnostics.csv', index=False)
    summarize(pred, y)
    assert COUNTS == {'logistic_fits': 600, 'neural_optimizer_starts': 1200} and len(pred) == 6720
    dump('completion.json', {'status': 'COMPLETED', **COUNTS, 'environment_changed': False})
    dump('output_manifest.json', {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()})


def summarize(pred, y):
    metrics = []
    for (method, scenario, repeat), g in pred.groupby(['method', 'scenario', 'repeat']):
        p = g[['p0', 'p1', 'p2']].to_numpy(); target = g.y_true.to_numpy()
        metrics.append({'method': method, 'scenario': scenario, 'repeat': repeat,
            'balanced_accuracy': balanced_accuracy_score(target, p.argmax(axis=1)),
            'macro_f1': f1_score(target, p.argmax(axis=1), average='macro', zero_division=0),
            'log_loss': log_loss(target, p, labels=[0, 1, 2]), 'auc': roc_auc_score(target, p, multi_class='ovr', average='macro')})
    m = pd.DataFrame(metrics); m.to_csv(OUT / 'metrics_by_repeat.csv', index=False)
    summary = m.groupby(['method', 'scenario']).agg(balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        balanced_accuracy_std=('balanced_accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'),
        log_loss_mean=('log_loss', 'mean'), auc_mean=('auc', 'mean')).reset_index()
    summary.to_csv(OUT / 'performance_summary.csv', index=False)
    draws = [np.random.default_rng(SEED + 7000 + c).choice(np.flatnonzero(y == c),
                 size=(5000, (y == c).sum()), replace=True) for c in range(3)]
    contrasts = []
    for scenario in SCENARIOS:
        g = pred.loc[pred.scenario == scenario].assign(correct=lambda d: (d.y_true == d.y_pred).astype(float))
        correct = g.groupby(['participant_id', 'method']).correct.mean().unstack('method').sort_index()
        for baseline in ['Attention_MIL', 'DeepSets_mean', 'Median_LR', 'Mean_LR', 'Coherence_shuffled']:
            delta = (correct.Coherence_attention - correct[baseline]).to_numpy()
            bs = np.mean([delta[d].mean(axis=1) for d in draws], axis=0); lo, hi = np.quantile(bs, [.025, .975])
            contrasts.append({'scenario': scenario, 'baseline': baseline,
                'balanced_accuracy_delta': np.mean([delta[y == c].mean() for c in range(3)]),
                'conditional_ci_low': lo, 'conditional_ci_high': hi})
    contrasts = pd.DataFrame(contrasts); contrasts.to_csv(OUT / 'paired_participant_contrasts.csv', index=False)
    primary = contrasts.loc[(contrasts.scenario == 'One_mislinked_trial') & (contrasts.baseline == 'Attention_MIL')].iloc[0]
    clean = contrasts.loc[(contrasts.scenario == 'Clean') & (contrasts.baseline == 'Attention_MIL')].iloc[0]
    controls = contrasts.loc[(contrasts.scenario == 'One_mislinked_trial') & contrasts.baseline.isin(['Median_LR', 'Coherence_shuffled'])]
    passed = primary.balanced_accuracy_delta >= .03 and primary.conditional_ci_low > 0 and clean.conditional_ci_low > -.02 and (controls.conditional_ci_low > 0).all()
    dump('decision.json', {'status': 'INTERNAL_STRESS_TEST_SIGNAL_ONLY' if passed else 'FEASIBILITY_GATE_NOT_MET',
        'novelty_established': False, 'external_validation': False,
        'optimizer_starts_with_gradient_above_1e_3': int(sum(r['gradient_max'] > 1e-3 for r in OPTIMIZATION))})
    print(summary.loc[summary.scenario.isin(['Clean', 'One_mislinked_trial'])].to_string(index=False))
    print(contrasts.loc[(contrasts.baseline == 'Attention_MIL') | ((contrasts.scenario == 'One_mislinked_trial') & contrasts.baseline.isin(['Median_LR', 'Coherence_shuffled']))].to_string(index=False))
    print('Continuation gate:', passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        run()
