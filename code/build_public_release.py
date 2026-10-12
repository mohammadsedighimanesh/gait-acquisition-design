"""Assemble the public repository folder (code, protocols, derived results, manuscript). Excludes raw data, feature caches and fitted-model files.
CSV files larger than 20 MB are gzip-compressed (.csv.gz). Writes RELEASE_MANIFEST.json with SHA-256 of every released file."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REL = ROOT / 'release_gait_acquisition_design'
I = ROOT / 'idea1_2026'
GZ_LIMIT = 20 * 1024 * 1024
SKIP_DIRS = {'models', 'sources', 'influence_models', '__pycache__'}
SKIP_SUFFIX = {'.npz', '.h5', '.mat', '.xls', '.xlsx', '.pkl', '.zip'}
skipped = []; written = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.csv' and src.stat().st_size > GZ_LIMIT:
        dst = dst.with_suffix('.csv.gz')
        with open(src, 'rb') as fi, gzip.open(dst, 'wb', compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy2(src, dst)
    written.append(dst)


def copy_tree(src, dst):
    for p in sorted(src.rglob('*')):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts) or p.suffix.lower() in SKIP_SUFFIX:
            skipped.append(str(p.relative_to(ROOT)).replace('\\', '/')); continue
        copy_file(p, dst / rel)


if REL.exists():
    raise FileExistsError(REL)
REL.mkdir()
# code
for name in ['restricted_acquisition_pilot.py', 'restricted_acquisition_grid.py', 'restricted_acquisition_sensitivity.py', 'restricted_acquisition_gaitrec.py', 'restricted_acquisition_policy.py',
             'restricted_acquisition_stepup.py', 'restricted_acquisition_stepup_a52.py', 'download_stepup_third_task.py', 'verify_restricted_acquisition_pilot.py', 'verify_restricted_acquisition_grid.py',
             'verify_restricted_acquisition_sensitivity.py', 'verify_restricted_acquisition_gaitrec.py', 'verify_restricted_acquisition_policy.py', 'verify_restricted_acquisition_stepup.py',
             'verify_restricted_acquisition_stepup_a52.py', 'build_manuscript_assets.py', 'build_manuscript_docx.py', 'audit_manuscript_numbers.py', 'build_public_release.py',
             'corrected_raw_pipeline_diagnostic.py', 'learned_multitrial_aggregation_pilot.py', 'gaitrec_trial_stability_pilot.py', 'fixed_model_trial_budget_audit.py']:
    copy_file(ROOT / name, REL / 'code' / name)
# protocols, reports and results
copy_tree(I / 'restricted_acquisition_learning_surface_v1', REL / 'results' / 'fpi_gaitrec_and_protocols')
copy_tree(I / 'stepup_third_task_v1', REL / 'results' / 'stepup')
copy_tree(I / 'fixed_model_trial_budget_audit_v1', REL / 'results' / 'fpi_assessment_side_earlier')
for f in ['cohort_audit.json', 'protocol_before_fit.json', 'selected_development_sessions.csv', 'outer_person_splits_before_fit.csv', 'inner_tuning.csv']:
    copy_file(I / 'gaitrec_trial_stability_pilot_v1' / f, REL / 'results' / 'gaitrec_cohort_definition' / f)
for f in ['outer_splits_before_fit.csv', 'frozen_config.json']:
    copy_file(I / 'learned_multitrial_aggregation_pilot_v1' / f, REL / 'results' / 'fpi_cohort_definition' / f)
copy_file(I / 'internal_trial_budget_study_v1' / 'inner_tuning.csv', REL / 'results' / 'fpi_cohort_definition' / 'earlier_nested_inner_tuning.csv')
# manuscript
M = ROOT / 'Paper/Manuscript_Acquisition_Design_2026_10_12'
for name in ['Manuscript_final.docx', 'Supplementary_Information_final.docx', 'Manuscript_final_preview.pdf', 'Supplementary_Information_final_preview.pdf', 'manuscript.md', 'supplementary_information.md',
             'manuscript_template.md', 'claim_to_evidence_table.md', 'number_audit.txt', 'numbers.json']:
    copy_file(M / name, REL / 'manuscript' / name)
copy_tree(M / 'figures', REL / 'manuscript' / 'figures'); copy_tree(M / 'tables', REL / 'manuscript' / 'tables')
manifest = {str(p.relative_to(REL)).replace('\\', '/'): sha(p) for p in sorted(written)}
(REL / 'RELEASE_MANIFEST.json').write_text(json.dumps({'files': manifest, 'excluded_from_release': skipped}, indent=1), encoding='utf-8')
total = sum(p.stat().st_size for p in written)
print(f'files: {len(written)}  total MB: {total / 1e6:.1f}  excluded: {len(skipped)}')
print('largest:', sorted(((p.stat().st_size / 1e6, str(p.relative_to(REL))) for p in written), reverse=True)[:6])
