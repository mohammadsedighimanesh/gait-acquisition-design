"""Independent verifier for A5 StepUP stages. Usage: python verify_restricted_acquisition_stepup.py build|pilot|surface|policy|sensitivity [L1|L2]. No refits."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import restricted_acquisition_stepup as g

TASKDIR = g.TASKDIR; STUDY = g.STUDY


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def replay(m, feats):
    z = (feats - m['s1_mean']) / m['s1_scale']; z = ((z - m['pca_mean']) @ m['pca_components'].T - m['s2_mean']) / m['s2_scale']; logits = z @ m['coef'].T + m['intercept']
    if len(m['classes']) == 1:
        p = np.ones((len(feats), 1))  # A5.2.2 constant model
    elif m['coef'].shape[0] == 1:
        p1 = 1 / (1 + np.exp(-logits[:, 0])); p = np.stack([1 - p1, p1], 1)
    else:
        p = softmax(logits)
    if len(m['classes']) < 2:
        full = np.zeros((len(feats), 2)); full[:, m['classes']] = p; p = full
    return p


def check_build(ch):
    bi = json.loads((TASKDIR / 'build/build_info.json').read_text()); c = np.load(TASKDIR / 'build/features_cache.npz')
    ch['features_cache_hash'] = sha(TASKDIR / 'build/features_cache.npz') == bi['features_cache_sha256']
    ch['source_hashes_unchanged'] = all(sha(ROOT / 'idea1_2026' / ('stepup_third_task_v1/sources/' + k) if (TASKDIR / 'sources' / k).exists() else next(s / k for s in g.SOURCES if (s / k).exists())) == h for k, h in bi['source_file_sha256'].items())
    audit = pd.read_csv(TASKDIR / 'build/eligibility_audit.csv'); inc = audit[audit.status == 'INCLUDED']
    ch['included_matches_cache'] = len(inc) == len(c['person_ids']) and set(inc.pid.astype(int)) == set(c['person_ids'].astype(int))
    rng = np.random.default_rng(1); ok = True; maxerr = 0.0
    for _, r in inc.sample(min(25, len(inc)), random_state=1).iterrows():
        d = g.locate(f'{int(r.pid):03d}'); meta = pd.read_csv(d / 'metadata.csv'); rows, fsid = g.eligible_steps(meta)
        ok &= ','.join(map(str, rows[:g.KMAX])) == r.bag_row_indices and ','.join(map(str, fsid[:g.KMAX])) == r.bag_footstep_ids
        arr = np.load(d / 'pipeline_2.npz')['arr_0']; j = int(rng.integers(g.KMAX)); f = g.step_features(arr[rows[j]])
        maxerr = max(maxerr, float(np.abs(f - c['X'][int(np.where(c['person_ids'] == int(r.pid))[0][0]), j]).max()))
    ch['eligibility_and_bag_replayed'] = bool(ok); ch['feature_replay_max_error'] = maxerr; ch['feature_replay_within_1e-12'] = maxerr < 1e-12
    pm = pd.read_csv(g.META); pm = pm.set_index(pm.ParticipantID.astype(int))
    ch['labels_match_metadata'] = bool(((pm.loc[c['person_ids'], 'Sex'] == 'Male').astype(int).to_numpy() == c['sex']).all())
    for L in ['L1', 'L2']:
        s = pd.read_csv(TASKDIR / f'build/outer_splits_{L}.csv'); ok2 = True
        for (r, f), gg in s.groupby(['repeat', 'fold']):
            ok2 &= not set(gg[gg.role == 'train'].person_id) & set(gg[gg.role == 'test'].person_id) and set(gg.person_id) == set(c['person_ids'].astype(int))
        ch[f'splits_{L}_valid'] = bool(ok2)


def check_surface(ch, out, L, cells, draws):
    fr = json.loads((STUDY / 'frozen_protocol_A5.json').read_text())
    ch['addendum_hash'] = sha(STUDY / 'ADDENDUM_A5_STEPUP_THIRD_TASK.md') == fr['addendum_A5_sha256']; ch['script_hash'] = sha(ROOT / 'restricted_acquisition_stepup.py') == fr['script_sha256']
    man = json.loads((out / 'output_manifest.json').read_text()); ch['manifest_all_hashed'] = all(sha(out / n) == h for n, h in man.items())
    prog = json.loads((out / 'progress.json').read_text()); task = g.Task(L); X = task.X; y = task.y
    mem = pd.read_csv(out / 'acquisition_memberships.csv'); pred = pd.read_csv(out / 'heldout_predictions.csv'); audit = pd.read_csv(out / 'acquisition_audit.csv'); conv = pd.read_csv(out / 'convergence_audit.csv')
    nfits = prog['primary_fits']; ch['fit_counts'] = nfits == prog['exclusion_control_fits'] == len(audit) == len(conv) == sum(n.startswith('models/') for n in man)
    ch['exclusion_flags_all_true'] = bool(audit.exclusion_proof_identical.all()); ch['convergence_no_max_iter'] = int(conv.reached_max_iter.sum()) == 0
    alloc_by_fold = {k: {int(n): tuple(c) for n, c in v.items()} for k, v in prog['allocation_by_fold'].items()}
    train = mem[mem.role == 'train']; test = mem[mem.role == 'test']; nested = True; classes_ok = True; disjoint = True; replayerr = 0.0; excl = 0.0; models = 0
    for (r, f), gd0 in train.groupby(['repeat', 'fold']):
        blk = task.splits[(task.splits.repeat == r) & (task.splits.fold == f)]; te_rec = set(blk[blk.role == 'test'].person_id); tr_rec = set(blk[blk.role == 'train'].person_id)
        disjoint &= set(test[(test.repeat == r) & (test.fold == f)].person_id) == te_rec; alloc = alloc_by_fold[f'{r}_{f}']
        tr_counts = np.bincount(y[np.array(sorted(task.pos[int(p)] for p in tr_rec))], minlength=2); classes_ok &= all(alloc[n] == g.alloc_for(n, tr_counts) for n in alloc)
        te_idx = np.array(sorted(task.pos[int(p)] for p in te_rec)); tf = task.test_feats(te_idx)
        for d, gd in gd0.groupby('draw'):
            ns_here = sorted(gd.n.unique()); prev_people = None
            for n in ns_here:
                prev_trials = None
                for k in sorted(gd[gd.n == n].k.unique()):
                    cell = gd[(gd.n == n) & (gd.k == k)]; people = set(cell.person_id)
                    classes_ok &= tuple(np.bincount(cell['class'].to_numpy(), minlength=2)) == alloc[n] and cell['class'].nunique() == 2; disjoint &= not people & te_rec and people <= tr_rec
                    trials = {int(p): set(map(int, t.split(','))) for p, t in zip(cell.person_index, cell.retained_trials)}; nested &= all(len(t) == k for t in trials.values())
                    if prev_trials is not None:
                        nested &= all(prev_trials[p] <= trials[p] for p in trials)
                    prev_trials = trials
                    m = np.load(out / 'models' / f'r{r}_f{f}_d{d}_n{n}_k{k}.npz'); models += 1
                    trf = np.stack([np.median(X[p, sorted(np.array(list(trials[p])) - 1)], axis=0) for p in sorted(trials)]); sd = trf.std(0)
                    excl = max(excl, float(np.abs(trf.mean(0) - m['s1_mean']).max()), float(np.abs(np.where(sd > 0, sd, 1.0) - m['s1_scale']).max()))
                    saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.draw == d) & (pred.n == n) & (pred.k == k)].sort_values('person_index')
                    replayerr = max(replayerr, float(np.abs(replay(m, tf) - saved[['p0', 'p1']].to_numpy()).max()))
                if prev_people is not None:
                    nested &= prev_people <= set(gd[gd.n == n].person_id)
                prev_people = set(gd[gd.n == n].person_id)
    ch.update({'nested_people_and_trials': bool(nested), 'class_counts_match': bool(classes_ok), 'test_recorded_and_disjoint': bool(disjoint), 'models_replayed': models,
               'max_probability_replay_error': replayerr, 'replay_within_1e-8': replayerr < 1e-8, 'scaler1_from_retained_trials_max_error': excl, 'exclusion_within_1e-8': excl < 1e-8})
    s, c, _, _ = g.summarize_surface(pred, y, cells, g.CONTRASTS if len(cells) > 2 else [g.CONTRASTS[0]], draws); s0 = pd.read_csv(out / 'learning_surface.csv'); c0 = pd.read_csv(out / 'prespecified_contrasts.csv')
    ch['surface_recomputed_match'] = bool(np.allclose(s[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(), s0[['balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'log_loss']].to_numpy(), atol=1e-12))
    ch['contrast_recomputed_match'] = bool(np.allclose(c[['mean', 'ci_low', 'ci_high']].to_numpy(), c0[['mean', 'ci_low', 'ci_high']].to_numpy(), atol=1e-12))


def check_policy(ch, out, L):
    fr = json.loads((STUDY / 'frozen_protocol_A5.json').read_text()); ch['script_hash'] = sha(ROOT / 'restricted_acquisition_stepup.py') == fr['script_sha256']
    man = json.loads((out / 'output_manifest.json').read_text()); ch['manifest_all_hashed'] = all(sha(out / n) == h for n, h in man.items())
    prog = json.loads((out / 'progress.json').read_text()); ch['final_fits_equal_models'] = prog['final_fits'] == sum(n.startswith('models/') for n in man) == prog['exclusion_control_fits']
    task = g.Task(L); acq = pd.read_csv(out / 'acquisitions.csv'); pred = pd.read_csv(out / 'heldout_predictions.csv'); dec = pd.read_csv(out / 'policy_decisions.csv')
    acq_ok = True; replayerr = 0.0; excl = 0.0; models = 0; dec_ok = True; ties = 0
    for (r, f, s, B, pol), ga in acq.groupby(['repeat', 'fold', 'seed', 'budget', 'policy']):
        blk = task.splits[(task.splits.repeat == r) & (task.splits.fold == f)]; tr = {task.pos[int(p)] for p in blk[blk.role == 'train'].person_id}; te = {task.pos[int(p)] for p in blk[blk.role == 'test'].person_id}
        acq_ok &= ga.n_trials.sum() == B and set(ga.person_index) <= tr and not set(ga.person_index) & te and ga.n_trials.max() <= g.KMAX and ga.person_index.is_unique
        m = np.load(out / 'models' / f'r{r}_f{f}_s{s}_B{B}_{pol}.npz'); models += 1
        trf = np.stack([np.median(task.X[int(p), np.array([int(t) - 1 for t in tt.split(',')])], axis=0) for p, tt in zip(ga.person_index, ga.trials)]); sd = trf.std(0)
        excl = max(excl, float(np.abs(trf.mean(0) - m['s1_mean']).max()), float(np.abs(np.where(sd > 0, sd, 1.0) - m['s1_scale']).max()))
        tf = task.test_feats(np.array(sorted(te))); saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.seed == s) & (pred.budget == B) & (pred.policy == pol)].sort_values('person_index')
        replayerr = max(replayerr, float(np.abs(replay(m, tf) - saved[['p0', 'p1']].to_numpy()).max()))
    md = dec[dec.policy == 'MARGIN']
    for key, gd in md.groupby(['repeat', 'fold', 'seed', 'budget', 'step']):
        ch_ = gd[gd.replicate == gd.chosen_replicate.iloc[0]].iloc[0]
        if np.isnan(ch_.gain_people) or np.isnan(ch_.gain_trial):
            dec_ok &= ch_.action == 'people' and bool(ch_.undefined_estimate)
        elif abs(ch_.gain_people - ch_.gain_trial) < 1e-9:
            ties += 1
        else:
            dec_ok &= ('people' if ch_.gain_people >= ch_.gain_trial else 'trials') == ch_.action
    rec = g.summarize_policy(pred, task.y, 2, g.POLICIES, g.POLICY_BUDGETS); s0 = pd.read_csv(out / 'policy_summary.csv')
    ch.update({'acquisition_logs_valid': bool(acq_ok), 'margin_decisions_reproduced': bool(dec_ok), 'csv_precision_ties_accepted': ties, 'models_replayed': models, 'max_probability_replay_error': replayerr,
               'replay_within_1e-8': replayerr < 1e-8, 'scaler_from_acquired_trials_max_error': excl, 'exclusion_within_1e-8': excl < 1e-8,
               'summary_recomputed_match': bool(np.allclose(rec[['mean', 'ci_low', 'ci_high']].to_numpy(), s0[['mean', 'ci_low', 'ci_high']].to_numpy(), atol=1e-12, equal_nan=True))})


def check_sensitivity(ch, out, L):
    fr = json.loads((STUDY / 'frozen_protocol_A5.json').read_text()); ch['script_hash'] = sha(ROOT / 'restricted_acquisition_stepup.py') == fr['script_sha256']
    man = json.loads((out / 'output_manifest.json').read_text()); ch['manifest_all_hashed'] = all(sha(out / n) == h for n, h in man.items())
    inner = pd.read_csv(out / 'inner_tuning_losses.csv'); sel = pd.read_csv(out / 'selected_C.csv'); ok = True
    for _, r in sel.iterrows():
        gg = inner[(inner.repeat == r['repeat']) & (inner.fold == r['fold']) & (inner.draw == r['draw']) & (inner.n == r['n']) & (inner.k == r['k'])]; means = gg.groupby('C').inner_val_log_loss.mean()
        ok &= len(gg) == 9 and min([0.1, 1.0, 10.0], key=lambda C: (means[C], C)) == r['selected_C']
    ch['selection_rule_replayed'] = bool(ok)
    task = g.Task(L); pred = pd.read_csv(out / 'heldout_predictions_tuned.csv'); mem = pd.read_csv(TASKDIR / f'surface_{L}_v1/acquisition_memberships.csv'); replayerr = 0.0; models = 0
    for (r, f, d, n, k), gp in pred.groupby(['repeat', 'fold', 'draw', 'n', 'k']):
        m = np.load(out / 'models' / f'r{r}_f{f}_d{d}_n{n}_k{k}.npz'); models += 1; te = np.sort(gp.person_index.to_numpy()); tf = task.test_feats(te)
        replayerr = max(replayerr, float(np.abs(replay(m, tf) - gp.sort_values('person_index')[['p0', 'p1']].to_numpy()).max()))
    ch.update({'models_replayed': models, 'max_probability_replay_error': replayerr, 'replay_within_1e-8': replayerr < 1e-8})


def main(stage, L):
    ch = {}
    if stage == 'build':
        check_build(ch); out = TASKDIR / 'build'
    elif stage == 'pilot':
        out = TASKDIR / 'pilot_v1'; check_surface(ch, out, 'L1', [(120, 5), (60, 10)], 5)
    elif stage == 'surface':
        out = TASKDIR / f'surface_{L}_v1'; check_surface(ch, out, L, [(n, k) for n in g.NS for k in g.KS], g.DRAWS)
    elif stage == 'policy':
        out = TASKDIR / f'policy_{L}_v1'; check_policy(ch, out, L)
    else:
        out = TASKDIR / f'sensitivity_{L}_v1'; check_sensitivity(ch, out, L)
    bools = {k: v for k, v in ch.items() if isinstance(v, bool)}
    rep = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'stage': stage, 'label': L, 'checks': ch, 'verification_model_fits': 0}
    (out / 'independent_verification.json').write_text(json.dumps(rep, indent=2, default=float)); print(json.dumps(rep, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'L1')
