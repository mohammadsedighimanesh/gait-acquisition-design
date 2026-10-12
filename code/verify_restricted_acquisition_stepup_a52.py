"""Verifier for A5.2 (reuses the A5 policy checks with the A5.2 budgets). Usage: python verify_restricted_acquisition_stepup_a52.py L1|L2. No refits."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import restricted_acquisition_stepup as g
import verify_restricted_acquisition_stepup as V


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(L):
    out = g.TASKDIR / f'policy_a52_{L}_v1'; fr = json.loads((g.STUDY / 'frozen_protocol_A5_2.json').read_text()); ch = {}
    ch['addendum_hash'] = sha(g.STUDY / 'ADDENDUM_A5_2_SMALL_BUDGET_POLICY.md') == fr['addendum_A5_2_sha256']
    ch['a52_script_hash'] = sha(ROOT / 'restricted_acquisition_stepup_a52.py') == fr['script_sha256']
    ch['stepup_module_hash'] = sha(ROOT / 'restricted_acquisition_stepup.py') == fr['stepup_module_sha256']
    g.POLICY_BUDGETS = fr['budgets']  # so the A5 policy checker recomputes with the A5.2 budget order
    V.check_policy(ch, out, L); ch.pop('script_hash', None)
    bools = {k: v for k, v in ch.items() if isinstance(v, bool)}
    rep = {'status': 'PASS_LISTED_CHECKS_ONLY' if all(bools.values()) else 'FAIL', 'stage': 'policy_a52', 'label': L, 'checks': ch, 'verification_model_fits': 0}
    (out / 'independent_verification.json').write_text(json.dumps(rep, indent=2, default=float)); print(json.dumps(rep, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
