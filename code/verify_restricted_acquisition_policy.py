"""Independent verifier for policy_pilot_v1 / policy_grid_v1 (Addendum A4). No refits. Usage: python verify_restricted_acquisition_policy.py pilot|grid"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'idea1_2026/restricted_acquisition_learning_surface_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main(stage):
    import restricted_acquisition_policy as g
    out = STUDY / f'policy_{stage}_v1'; ch = {}
    fr = json.loads((STUDY / 'frozen_protocol_A4.json').read_text())
    ch['addendum_hash'] = sha(STUDY / 'ADDENDUM_A4_SEQUENTIAL_POLICY.md') == fr['addendum_A4_sha256']
    ch['script_hash'] = sha(ROOT / 'restricted_acquisition_policy.py') == fr['script_sha256']
    man = json.loads((out / 'output_manifest.json').read_text()); ch['manifest_all_hashed'] = all(sha(out / n) == h for n, h in man.items())
    prog = json.loads((out / 'progress.json').read_text()); ch['convergence_warnings'] = prog['convergence_warnings']
    ch['final_fits_equal_models'] = prog['final_fits'] == sum(n.startswith('models/') for n in man) and prog['exclusion_control_fits'] == prog['final_fits']
    tasks = ['FPI'] if stage == 'pilot' else ['FPI', 'GaitRec']
    replay = 0.0; excl = 0.0; models = 0; acq_ok = True; dec_ok = True; summ_ok = True; ties = 0
    for tname in tasks:
        task = g.Task(tname); acq = pd.read_csv(out / f'{tname}_acquisitions.csv'); pred = pd.read_csv(out / f'{tname}_heldout_predictions.csv')
        dec = pd.read_csv(out / f'{tname}_policy_decisions.csv')
        for (r, f, s, B, pol), ga in acq.groupby(['repeat', 'fold', 'seed', 'budget', 'policy']):
            block = task.splits[(task.splits.repeat == r) & (task.splits.fold == f)]
            tr = {task.pos[int(p)] for p in block.loc[block.role == 'train', task.cfg['id_col']]}; te = {task.pos[int(p)] for p in block.loc[block.role == 'test', task.cfg['id_col']]}
            acq_ok &= ga.n_trials.sum() == B and set(ga.person_index) <= tr and not set(ga.person_index) & te and ga.n_trials.max() <= 5
            acq_ok &= all(len(set(t.split(','))) == n for t, n in zip(ga.trials, ga.n_trials)) and ga.person_index.is_unique
            # replay final model
            m = np.load(out / 'models' / f'{tname}_r{r}_f{f}_s{s}_B{B}_{pol}.npz'); models += 1
            trf = np.stack([task.person_vec(task.X, int(p), np.array([int(t) - 1 for t in tt.split(',')])) for p, tt in zip(ga.person_index, ga.trials)])
            sd = trf.std(0); excl = max(excl, float(np.abs(trf.mean(0) - m['s1_mean']).max()), float(np.abs(np.where(sd > 0, sd, 1.0) - m['s1_scale']).max()))
            te_sorted = np.array(sorted(te)); tf = task.test_feats(te_sorted); z = (tf - m['s1_mean']) / m['s1_scale']
            if 'pca_mean' in m:
                z = ((z - m['pca_mean']) @ m['pca_components'].T - m['s2_mean']) / m['s2_scale']
            logits = z @ m['coef'].T + m['intercept']
            if m['coef'].shape[0] == 1:  # binary sklearn convention when only two classes were acquired
                logits = np.concatenate([-logits, logits], axis=1) / 2  # not used for 3+ classes; kept for completeness
            p_hat = softmax(logits)
            if len(m['classes']) < task.nc:
                full = np.zeros((p_hat.shape[0], task.nc)); full[:, m['classes']] = p_hat; p_hat = full
            saved = pred[(pred.repeat == r) & (pred.fold == f) & (pred.seed == s) & (pred.budget == B) & (pred.policy == pol)].sort_values('person_index')
            replay = max(replay, float(np.abs(p_hat - saved[[f'p{c}' for c in range(task.nc)]].to_numpy()).max()))
        # MARGIN decisions reproducible from logged estimates and chosen replicate
        md = dec[dec.policy == 'MARGIN']
        for (r, f, s, B, step), gd in md.groupby(['repeat', 'fold', 'seed', 'budget', 'step']):
            chosen = gd[gd.replicate == gd.chosen_replicate.iloc[0]].iloc[0]
            if np.isnan(chosen.gain_people) or np.isnan(chosen.gain_trial):
                dec_ok &= chosen.action == 'people' and bool(chosen.undefined_estimate)
            elif abs(chosen.gain_people - chosen.gain_trial) < 1e-9:
                ties += 1  # CSV-precision tie (|gain difference| < 1e-9): the in-run float comparison decided; either action accepted
                dec_ok &= len(gd) == 3
            else:
                dec_ok &= (('people' if chosen.gain_people >= chosen.gain_trial else 'trials') == chosen.action) and len(gd) == 3
        budgets = g.TASKS[tname]['budgets']  # same order as the run (summary rows are ordered by this list)
        rec = g.summarize(pred, task.y, task.nc, g.POLICIES, budgets); s0 = pd.read_csv(out / f'{tname}_policy_summary.csv')
        summ_ok &= bool(np.allclose(rec[['mean', 'ci_low', 'ci_high']].to_numpy(), s0[['mean', 'ci_low', 'ci_high']].to_numpy(), atol=1e-12, equal_nan=True))
    ch.update({'acquisition_logs_valid': bool(acq_ok), 'margin_decisions_reproduced': bool(dec_ok), 'csv_precision_ties_accepted': int(ties), 'models_replayed': models, 'max_probability_replay_error': replay,
               'replay_within_1e-8': replay < 1e-8, 'scaler_from_acquired_trials_max_error': excl, 'exclusion_within_1e-8': excl < 1e-8, 'summary_recomputed_match': bool(summ_ok)})
    bools = {k: v for k, v in ch.items() if isinstance(v, bool)}
    rep = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'stage': stage, 'checks': ch, 'verification_model_fits': 0}
    (out / 'independent_verification.json').write_text(json.dumps(rep, indent=2, default=float)); print(json.dumps(rep, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
