"""Isolate inference-budget sensitivity without retraining or selecting new models."""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
import exhaustive_trial_budget_study as b

OUT = b.ROOT / 'idea1_2026/fixed_model_trial_budget_audit_v1'
SEED = 20261022


def run():
    if OUT.exists():
        raise FileExistsError(f'Existing outputs preserved: {OUT}')
    OUT.mkdir()
    cache_path = b.SOURCE / 'source_feature_cache.npz'
    split_path = b.SOURCE / 'outer_splits_before_fit.csv'
    models = [b.OUT / 'models' / f'r{r}_f{f}_{m}_k5.npz' for r in range(3) for f in range(5) for m in ['Mean', 'Median']]
    config = {'question': 'Does inference-trial-count sensitivity remain when the fitted model is unchanged?',
        'frozen_training_budget': 5, 'inference_budgets': [1, 2, 3, 4, 5], 'methods': ['Mean', 'Median'],
        'subset_selection': 'all combinations with uniform acquisition expectation; no probability ensembling',
        'bootstrap': '5000 class-stratified people, retaining repeat/subset exposure; conditional, no refit',
        'contrasts': ['five minus one', 'three minus five'], 'seed': SEED,
        'stage': 'post hoc design-confound sensitivity, not new independent confirmation or primary redefinition',
        'training_and_preprocessing_changed': False, 'clinical_noninferiority_margin_established': False,
        'source_cache_sha256': b.sha(cache_path), 'source_split_sha256': b.sha(split_path),
        'script_sha256': b.sha(Path(__file__)),
        'model_hashes': {str(p.relative_to(b.OUT)): b.sha(p) for p in models}}
    (OUT / 'frozen_protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    cache = np.load(cache_path); X, A, y = cache['X'], cache['A'], cache['labels']
    splits = pd.read_csv(split_path); rows = []
    original = pd.read_csv(b.OUT / 'heldout_subset_predictions.csv')
    for (r, f), g in splits.groupby(['repeat', 'fold']):
        tr = g.loc[g.role == 'train', 'participant_id'].to_numpy() - 1
        te = np.sort(g.loc[g.role == 'test', 'participant_id'].to_numpy()) - 1
        assert not set(tr) & set(te)
        for method in ['Mean', 'Median']:
            with np.load(b.OUT / 'models' / f'r{r}_f{f}_{method}_k5.npz') as state:
                np.testing.assert_allclose(state['x_mean'], X[tr].reshape(-1, 28).mean(axis=0), atol=1e-10)
                xx = (X[te] - state['x_mean']) / state['x_scale']
                aa = (A[te] - state['a_mean']) / state['a_scale']
                for k in range(1, 6):
                    subsets = list(itertools.combinations(range(5), k)); n = len(subsets)
                    features = b.features(xx, aa, subsets, method)
                    p = softmax(((features - state['pool_mean']) / state['pool_scale']) @ state['coef'].T + state['intercept'], axis=1).reshape(len(te), n, 3)
                    if k == 5:
                        saved = original.query('repeat == @r and fold == @f and method == @method and k == 5').sort_values('participant_id')
                        np.testing.assert_allclose(p[:, 0], saved[['p0', 'p1', 'p2']], atol=1e-10)
                    for i, pid in enumerate(te):
                        rows.extend({'repeat': r, 'fold': f, 'method': method, 'training_k': 5, 'k': k,
                            'participant_id': int(pid + 1), 'subset_id': j, 'trial_ids': ';'.join(str(t + 1) for t in sub),
                            'y_true': int(y[pid]), 'y_pred': int(q.argmax()), **dict(zip(['p0', 'p1', 'p2'], q))}
                            for j, (sub, q) in enumerate(zip(subsets, p[i])))
    pred = pd.DataFrame(rows); pred.to_csv(OUT / 'fixed_model_subset_predictions.csv', index=False)
    per = pd.DataFrame([{'method': m, 'k': k, 'repeat': r, **b.expected_metrics(g)}
        for (m, k, r), g in pred.groupby(['method', 'k', 'repeat'])])
    per.to_csv(OUT / 'fixed_model_metrics_by_repeat.csv', index=False)
    summary = per.groupby(['method', 'k']).mean(numeric_only=True).drop(columns='repeat').reset_index()
    old = pd.read_csv(b.OUT / 'trial_budget_summary.csv')[['method', 'k', 'expected_balanced_accuracy']]
    summary = summary.merge(old.rename(columns={'expected_balanced_accuracy': 'budget_specific_training_ba'}), on=['method', 'k'])
    summary['fixed_minus_budget_specific_ba'] = summary.expected_balanced_accuracy - summary.budget_specific_training_ba
    summary.to_csv(OUT / 'fixed_vs_budget_specific_summary.csv', index=False)
    persons = pred.assign(correct=(pred.y_true == pred.y_pred).astype(float)).groupby(['method', 'k', 'repeat', 'participant_id']).agg(
        correct_fraction=('correct', 'mean'), distinct_predictions=('y_pred', 'nunique'), true_class=('y_true', 'first')).reset_index()
    persons.to_csv(OUT / 'fixed_model_person_subset_sensitivity.csv', index=False)
    instability = persons.groupby(['method', 'k']).agg(fraction_person_repeat_any_disagreement=(
        'distinct_predictions', lambda x: (x > 1).mean())).reset_index()
    instability.to_csv(OUT / 'fixed_model_choice_instability.csv', index=False)
    draws = [np.random.default_rng(SEED + c).choice(np.flatnonzero(y == c), size=(5000, int(np.sum(y == c))), replace=True) for c in range(3)]
    values = persons.groupby(['method', 'k', 'participant_id']).correct_fraction.mean().unstack(['method', 'k']).sort_index()
    intervals, contrasts = [], []
    for method in ['Mean', 'Median']:
        for k in range(1, 6):
            v = values[(method, k)].to_numpy(); bs = np.mean([v[d].mean(axis=1) for d in draws], axis=0)
            lo, hi = np.quantile(bs, [.025, .975])
            intervals.append({'method': method, 'k': k, 'expected_ba': np.mean([v[y == c].mean() for c in range(3)]),
                              'conditional_ci_low': lo, 'conditional_ci_high': hi})
        for ka, kb in [(5, 1), (3, 5)]:
            delta = (values[(method, ka)] - values[(method, kb)]).to_numpy()
            bs = np.mean([delta[d].mean(axis=1) for d in draws], axis=0)
            lo, hi = np.quantile(bs, [.025, .975])
            contrasts.append({'method': method, 'inference_k_candidate': ka, 'inference_k_baseline': kb,
                'fixed_model_ba_delta': np.mean([delta[y == c].mean() for c in range(3)]), 'conditional_ci_low': lo, 'conditional_ci_high': hi})
    pd.DataFrame(intervals).to_csv(OUT / 'fixed_model_performance_ci.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'fixed_model_paired_contrasts.csv', index=False)
    assert len(pred) == 13020
    (OUT / 'completion.json').write_text(json.dumps({'status': 'COMPLETED', 'replayed_models': 30,
        'actual_new_classifier_fits': 0, 'prediction_rows': len(pred), 'unique_people': 70,
        'source_or_environment_modified': False}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False)); print(pd.DataFrame(contrasts).to_string(index=False)); print(instability.to_string(index=False))


if __name__ == '__main__':
    run()
