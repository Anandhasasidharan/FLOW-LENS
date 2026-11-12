"""Fetch the publisher's richer NetFlow v3 benchmark for FlowLens v2."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = 'abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f'
API = f'https://api.rdm.uq.edu.au/datasets/{IDENTIFIER}'


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def main():
    directory = ROOT / 'data/raw/v2'
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / 'NF-UNSW-NB15-v3.zip'
    details = get_json(API + '/file-details')
    if details['accessType'] != 'OPEN' or details['status'] != 'PUBLISHED':
        raise RuntimeError('Dataset is not available with open access.')
    if not archive_path.exists():
        download = get_json(API + '/download')
        if not download.get('canDownload'):
            raise RuntimeError('Public download unavailable.')
        partial = directory / 'download.partial'
        received = 0
        with urllib.request.urlopen(download['url'], timeout=180) as source, partial.open('wb') as target:
            while block := source.read(1024 * 1024):
                target.write(block)
                received += len(block)
                if received > 250 * 1024 * 1024:
                    raise RuntimeError('Archive exceeds expected size limit.')
                if received // (20 * 1024 * 1024) != (received - len(block)) // (20 * 1024 * 1024):
                    print(f'Downloaded {received // (1024 * 1024)} MiB', flush=True)
        partial.replace(archive_path)
    outputs = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith('.csv')]
        print('CSV members:', [(Path(m.filename).name, m.file_size) for m in members], flush=True)
        for member in members:
            if member.file_size > 3 * 1024 ** 3:
                raise RuntimeError('Unexpected uncompressed size.')
            destination = directory / Path(member.filename).name
            if not destination.exists() or destination.stat().st_size != member.file_size:
                with archive.open(member) as source, destination.open('wb') as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
            digest = hashlib.sha256()
            with destination.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            outputs.append({'name': destination.name, 'bytes': destination.stat().st_size, 'sha256': digest.hexdigest()})
    provenance = {
        'dataset': details['name'], 'source': f'https://rdm.uq.edu.au/files/{IDENTIFIER}',
        'description': details['description'], 'license': details['datasetLicense']['title'],
        'reference': 'https://staff.itee.uq.edu.au/marius/NIDS_datasets/',
        'files': outputs,
    }
    (ROOT / 'data/rich_dataset_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Rich dataset ready.', flush=True)


if __name__ == '__main__':
    main()
