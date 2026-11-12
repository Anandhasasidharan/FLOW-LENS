"""Create chronological development/lockbox partitions before training."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.interpretable_detector import IDENTIFIERS


def main():
    source = ROOT / 'data/raw/v2/NF-UNSW-NB15-v3.csv'
    out = ROOT / 'data/raw/v2/prepared'
    out.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / 'models/v2/evaluation_plan.json'
    existing = json.loads(plan_path.read_text()) if plan_path.exists() else None
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    source_hash = digest.hexdigest()
    if existing and existing['source_sha256'] != source_hash:
        raise ValueError('Source data differ from the recorded evaluation plan.')
    data = pd.read_csv(source)
    if not data.Label.isin([0, 1]).all():
        raise ValueError('Unexpected labels.')
    columns = [c for c in data.columns if c not in IDENTIFIERS]
    start = data.FLOW_START_MILLISECONDS.to_numpy()
    end = data.FLOW_END_MILLISECONDS.to_numpy()
    cutoffs = np.asarray(existing['cutoffs_ms']) if existing else np.quantile(start, [.60, .75, .85], method='higher')
    roles = {
        'train': (start < cutoffs[0]) & (end < cutoffs[0]),
        'validation': (start >= cutoffs[0]) & (start < cutoffs[1]) & (end < cutoffs[1]),
        'calibration': (start >= cutoffs[1]) & (start < cutoffs[2]) & (end < cutoffs[2]),
        'lockbox': start >= cutoffs[2],
    }
    counts = {}
    for role, mask in roles.items():
        part = data.loc[mask, columns + ['Label', 'Attack', 'FLOW_START_MILLISECONDS']].copy()
        part.to_parquet(out / f'{role}.parquet', index=False)
        counts[role] = {'records': len(part), 'start_ms': int(part.FLOW_START_MILLISECONDS.min()), 'end_ms': int(data.loc[mask, 'FLOW_END_MILLISECONDS'].max())}
        if role != 'lockbox':
            counts[role]['attack_counts'] = part.Attack.value_counts().to_dict()
        print(role, counts[role], flush=True)
    used = np.logical_or.reduce(list(roles.values()))
    plan = {
        'dataset': 'NF-UNSW-NB15-v3', 'source_records': len(data), 'features': columns,
        'identifier_columns_excluded': sorted(IDENTIFIERS),
        'split': 'Chronological 60/15/10/15 percent by flow-start quantiles, purged at boundaries using flow-end timestamps.',
        'cutoffs_ms': cutoffs.astype('int64').tolist(), 'partitions': counts,
        'boundary_flows_purged': int((~used).sum()),
        'selection': 'Development only. Prefer native EBM if within 1 percentage point validation average precision and family-balanced detection utility of challenger. Select threshold for macro attack-family recall at <=1% validation benign FPR.',
        'final_test_rule': 'Lockbox predictions are computed once after models, probability policies, and thresholds are frozen. A failed result does not trigger tuning on the lockbox.',
        'source_sha256': source_hash,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        if existing['features'] != columns or existing['partitions'] != counts:
            raise ValueError('Prepared partitions differ from the recorded evaluation plan.')
        print('Prepared data reconstructed; existing evaluation plan unchanged.', flush=True)
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
        print('Evaluation plan locked.', flush=True)


if __name__ == '__main__':
    main()
