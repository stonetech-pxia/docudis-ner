"""Download the NER model binaries listed in models/manifest.json.

    "$PYTHON" tool/fetch_models.py                 # the shipped model, xlmr_ner_docudis
    "$PYTHON" tool/fetch_models.py xlmr_ner_hrl    # named models (the stock ones are for A/B benchmarks)
    "$PYTHON" tool/fetch_models.py --all
    "$PYTHON" tool/fetch_models.py --dest ../app/assets/models   # model.json and binaries for an app checkout

Needs `pip install huggingface_hub`. xlmr_ner_docudis is a private repo: run
`huggingface-cli login` once first. Files already present with the right SHA-256
are skipped; a download with the wrong hash is refused.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = Path(__file__).resolve().parent.parent / 'models'
DEFAULT = 'xlmr_ner_docudis'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((MODELS / 'manifest.json').read_text(encoding='utf-8'))['models']
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', metavar='name',
                        help=f'models to fetch ({", ".join(manifest)}); default {DEFAULT}')
    parser.add_argument('--all', action='store_true', help='fetch every model in the manifest')
    parser.add_argument('--dest', type=Path, default=MODELS,
                        help='directory that receives <name>/model.json and the binaries (default: models/)')
    args = parser.parse_args()
    names = list(manifest) if args.all else args.names or [DEFAULT]
    unknown = [n for n in names if n not in manifest]
    if unknown:
        parser.error(f'not in the manifest: {", ".join(unknown)}')

    failed = False
    for name in names:
        model = manifest[name]
        if args.dest.resolve() != MODELS.resolve():
            (args.dest / name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MODELS / name / 'model.json', args.dest / name / 'model.json')
        for dest, spec in model['files'].items():
            target = args.dest / name / dest
            if target.exists() and sha256(target) == spec['sha256']:
                print(f'{name}/{dest}: up to date')
                continue
            print(f'{name}/{dest}: downloading {model["repo"]}/{spec["path"]}')
            cached = hf_hub_download(model['repo'], spec['path'], revision=model['revision'])
            if sha256(cached) != spec['sha256']:
                print(f'{name}/{dest}: SHA-256 mismatch, not installed', file=sys.stderr)
                failed = True
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
