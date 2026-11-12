"""Download NF-ToN-IoT directly from the University of Queensland (academic use)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = '2ad93cd0-ef9c-11ed-827d-e762de186848'
API = f'https://api.rdm.uq.edu.au/datasets/{DATASET_ID}'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['nf-ton-iot', 'nf-bot-iot'], default='nf-ton-iot')
    args = parser.parse_args()
    name, identifier = {
        'nf-ton-iot': ('NF-ToN-IoT', DATASET_ID),
        'nf-bot-iot': ('NF-BoT-IoT', '48fa1900-ef9c-11ed-827d-e762de186848'),
    }[args.dataset]
    api = f'https://api.rdm.uq.edu.au/datasets/{identifier}'
    destination = ROOT / f'data/raw/{name}.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    details = get_json(api + '/file-details')
    if details['accessType'] != 'OPEN' or details['status'] != 'PUBLISHED':
        raise RuntimeError('The dataset is not published with open access.')
    download = get_json(api + '/download')
    if not download.get('canDownload'):
        raise RuntimeError('The dataset is not available for public download.')
    # The short-lived download URL stays in memory; record only the stable source.
    with urllib.request.urlopen(download['url'], timeout=120) as response:
        archive_bytes = response.read(100 * 1024 * 1024 + 1)
    if len(archive_bytes) > 100 * 1024 * 1024:
        raise RuntimeError('Download exceeds the expected archive size limit.')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        files = [p for p in archive.infolist() if Path(p.filename).name == f'{name}.csv']
        print('CSV members:', [(p.filename, p.file_size) for p in files])
        if len(files) != 1 or files[0].file_size > 500 * 1024 * 1024:
            raise RuntimeError('Unexpected dataset archive layout or size.')
        for member in archive.infolist():
            if Path(member.filename).name == 'NetFlow_v1_Features.csv':
                (ROOT / 'data/netflow_v1_features.csv').write_bytes(archive.read(member))
        with archive.open(files[0]) as source, destination.open('wb') as target:
            while block := source.read(1024 * 1024):
                target.write(block)
    provenance = {
        'dataset': details['name'],
        'source': f'https://rdm.uq.edu.au/files/{identifier}',
        'reference_page': 'https://staff.itee.uq.edu.au/marius/NIDS_datasets/',
        'license': details['datasetLicense']['title'],
        'citation': 'Sarhan, Layeghy, Moustafa, Portmann (2021). NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems.',
        'archive_sha256': hashlib.sha256(archive_bytes).hexdigest(),
        'csv_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'archive_bytes': len(archive_bytes),
        'csv_bytes': destination.stat().st_size,
    }
    (ROOT / ('data/dataset_provenance.json' if args.dataset == 'nf-ton-iot' else 'data/external_dataset_provenance.json')).write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Downloaded {destination.name}: {destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
