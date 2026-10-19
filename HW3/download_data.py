"""Fetch the supplied public course data and verify the recorded SHA256 hashes."""
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'starter/sources.json').read_text())
    for item in manifest:
        if not item['path'].startswith('data/'):
            continue
        target = root / item['path']
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], target
            print('Verified:', item['path'])
            continue
        url = 'https://drive.google.com/uc?export=download&id=' + item['drive_id']
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise RuntimeError(f"Unexpected download for {item['path']}; retrieve the original file from the Drive link in starter/sources.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print('Downloaded:', item['path'])


if __name__ == '__main__':
    main()
