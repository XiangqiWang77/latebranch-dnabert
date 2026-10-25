"""Verify downloaded study assets without loading model weights or executing them."""
import argparse,hashlib,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads((Path(__file__).parent/'asset_manifest.json').read_text())
    for name,expected in manifest.items():
        path=args.directory/name
        if path.stat().st_size!=expected['bytes']:raise ValueError('Size mismatch: '+name)
        checksum=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(4*1024*1024),b''):checksum.update(block)
        if checksum.hexdigest()!=expected['sha256']:raise ValueError('SHA-256 mismatch: '+name)
        print('verified',name,flush=True)

if __name__=='__main__':main()
