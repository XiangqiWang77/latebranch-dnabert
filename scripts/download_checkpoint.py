#!/usr/bin/env python3
"""Download the v0.1.0 adapted checkpoint and verify its published SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Destination directory')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/release_assets.json').read_text())
    record = manifest['assets']['latebranch-window.pt']
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'latebranch-window.pt'
    if destination.exists():
        if destination.stat().st_size != record['bytes'] or digest(destination) != record['sha256']:
            raise ValueError(f'Refusing to replace a different file: {destination}')
        print(f'Already verified: {destination}')
        return
    request = urllib.request.Request(record['url'], headers={'User-Agent': 'latebranch-dnabert/0.1.0'})
    temporary = None
    try:
        checksum = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=args.output, prefix='.checkpoint-', delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                for block in iter(lambda: response.read(8 << 20), b''):
                    checksum.update(block)
                    handle.write(block)
        if temporary.stat().st_size != record['bytes'] or checksum.hexdigest() != record['sha256']:
            raise ValueError('Downloaded checkpoint size or SHA-256 does not match the release manifest')
        temporary.replace(destination)
        print(f'Downloaded and verified: {destination}')
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
