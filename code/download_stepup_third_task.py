"""Download StepUP-P150 BF/W1 pipeline_2.npz + metadata.csv for the 108 participants not yet local, with checksum verification.
Checksums: official frdr-dfdr-checksums.txt. For .npz the published SHA256 refers to the uncompressed arr_0.npy payload (resolved earlier in this project);
for .csv it refers to the file bytes. Files go to idea1_2026/stepup_third_task_v1/sources/ (new, isolated). Nothing is executed from downloaded content."""
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'idea1_2026/stepup_third_task_v1/sources'
CHK = ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources/frdr-dfdr-checksums.txt'
BASE = 'https://www.frdr-dfdr.ca/repo/files/1/published/publication_1280/submitted_data/'
LOCAL = set(os.listdir(ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources/py')) | set(os.listdir(ROOT / 'idea1_2026/stepup_reliability_extension_v1/sources/py'))
TARGETS = [f'{i:03d}' for i in range(1, 151)]  # second pass: complete the full set in the task folder; verified local copies are copied, the rest downloaded
LOCAL_DIRS = [ROOT / 'idea1_2026/stepup_reliability_pilot_v1/sources', ROOT / 'idea1_2026/stepup_reliability_extension_v1/sources']


def published():
    text = CHK.read_text(encoding='utf-8', errors='replace'); out = {}
    for m in re.finditer(r'Filename/Fichier: (\S+)\nSHA256 Checksum/Sommes de contrôle: ([0-9a-f]{64})', text):
        out[m.group(1)] = m.group(2)
    return out


def payload_sha(npz_bytes):
    with zipfile.ZipFile(io.BytesIO(npz_bytes)) as z:
        return hashlib.sha256(z.read('arr_0.npy')).hexdigest()


def fetch(url, tries=8):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(10 * (t + 1))
    raise RuntimeError(f'{url}: {err}')


def main():
    pub = published(); OUT.mkdir(parents=True, exist_ok=True); log = []
    for pid in TARGETS:
        for name in ['pipeline_2.npz', 'metadata.csv']:
            rel = f'py/{pid}/BF/W1/{name}'; dest = OUT / rel
            if dest.exists():
                continue
            data = None; origin = 'download'
            for ld in LOCAL_DIRS:
                cand = ld / rel
                if cand.exists():
                    b = cand.read_bytes(); h = payload_sha(b) if name.endswith('.npz') else hashlib.sha256(b).hexdigest()
                    if h == pub.get(rel):
                        data = b; origin = f'local_copy:{ld.name}'; break
            if data is None:
                try:
                    data = fetch(BASE + rel)
                except Exception as e:
                    print(rel, 'FETCH_FAILED', repr(e)[:120], flush=True); log.append({'file': rel, 'verified': False, 'origin': 'fetch_failed'}); continue
            got = payload_sha(data) if name.endswith('.npz') else hashlib.sha256(data).hexdigest()
            ok = pub.get(rel) == got
            dest.parent.mkdir(parents=True, exist_ok=True)
            if ok:
                dest.write_bytes(data)
            log.append({'file': rel, 'bytes': len(data), 'published_sha256': pub.get(rel), 'observed_sha256': got, 'verified': ok, 'origin': origin})
            print(rel, len(data), origin, 'OK' if ok else 'MISMATCH', flush=True)
            if not ok:
                (dest.parent / (name + '.MISMATCH')).write_bytes(data)
    missing = [x for x in log if x.get('origin') == 'fetch_failed']
    if missing:
        print('FETCH_FAILED_COUNT', len(missing), flush=True)
    prev = OUT.parent / 'download_log.json'
    if prev.exists(): prev.rename(OUT.parent / 'download_log_pass1.json')
    (OUT.parent / 'download_log.json').write_text(json.dumps({'targets': TARGETS, 'n_targets': len(TARGETS), 'files': log,
                                                              'all_verified': all(x['verified'] for x in log)}, indent=2))
    print('done', len(log), 'files; all verified:', all(x['verified'] for x in log))


if __name__ == '__main__':
    main()
