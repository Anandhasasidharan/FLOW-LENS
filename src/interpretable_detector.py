"""Rich-flow feature engineering and native additive model explanations."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin

IDENTIFIERS = {
    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR', 'DNS_QUERY_ID', 'Label', 'Attack',
}
CATEGORICAL = {'PROTOCOL', 'L7_PROTO', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE'}
DERIVED = (
    'TOTAL_BYTES', 'TOTAL_PACKETS', 'BYTES_PER_PACKET', 'IN_BYTE_SHARE',
    'BYTE_DIRECTION_LOG_RATIO', 'PACKET_DIRECTION_LOG_RATIO', 'PACKETS_PER_SECOND', 'BYTES_PER_SECOND',
)


class RichFlowFeatures(TransformerMixin, BaseEstimator):
    """Fitted schema and category domains; feature engineering uses one row only."""
    def __init__(self, columns, exclude_ttl=False):
        self.columns = columns
        self.exclude_ttl = exclude_ttl

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(self.columns, dtype=object)
        self.n_features_in_ = len(self.columns)
        self.retained_ = [c for c in self.columns if not (self.exclude_ttl and c in {'MIN_TTL', 'MAX_TTL'})]
        self.categories_ = {c: sorted(pd.to_numeric(X[c], errors='coerce').dropna().unique()) for c in self.retained_ if c in CATEGORICAL}
        self.output_names_ = self.retained_ + list(DERIVED)
        return self

    def transform(self, X):
        missing = sorted(set(self.retained_) - set(X.columns))
        if missing:
            raise ValueError('Missing required rich-flow columns: ' + ', '.join(missing))
        result = X[self.retained_].apply(pd.to_numeric, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        for col in result:
            # Negative numeric network measurements denote unavailable/invalid observations.
            result.loc[result[col] < 0, col] = np.nan
        incoming = pd.to_numeric(X['IN_BYTES'], errors='coerce').clip(lower=0)
        outgoing = pd.to_numeric(X['OUT_BYTES'], errors='coerce').clip(lower=0)
        in_pkts = pd.to_numeric(X['IN_PKTS'], errors='coerce').clip(lower=0)
        out_pkts = pd.to_numeric(X['OUT_PKTS'], errors='coerce').clip(lower=0)
        duration = pd.to_numeric(X['FLOW_DURATION_MILLISECONDS'], errors='coerce').clip(lower=0) / 1000
        result['TOTAL_BYTES'] = incoming + outgoing
        result['TOTAL_PACKETS'] = in_pkts + out_pkts
        result['BYTES_PER_PACKET'] = result['TOTAL_BYTES'] / (result['TOTAL_PACKETS'] + 1)
        result['IN_BYTE_SHARE'] = incoming / (result['TOTAL_BYTES'] + 1)
        result['BYTE_DIRECTION_LOG_RATIO'] = np.log1p(incoming) - np.log1p(outgoing)
        result['PACKET_DIRECTION_LOG_RATIO'] = np.log1p(in_pkts) - np.log1p(out_pkts)
        result['PACKETS_PER_SECOND'] = result['TOTAL_PACKETS'] / (duration + .001)
        result['BYTES_PER_SECOND'] = result['TOTAL_BYTES'] / (duration + .001)
        result = result.replace([np.inf, -np.inf], np.nan)
        for col, values in self.categories_.items():
            result[col] = pd.Categorical(result[col], categories=values)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_names_, dtype=object)


def validate_rich_flow(raw, columns):
    if raw.empty or len(raw) > 100_000:
        raise ValueError('Provide between 1 and 100,000 records.')
    if not raw.columns.is_unique:
        raise ValueError('Duplicate column names are not supported.')
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError('Missing required rich-flow columns: ' + ', '.join(missing))
    clean = raw[list(columns)].apply(pd.to_numeric, errors='coerce')
    invalid_text = raw[list(columns)].notna() & clean.isna()
    if invalid_text.any().any():
        raise ValueError('Feature columns must contain numbers or empty cells.')
    if np.isinf(clean.to_numpy(dtype=float)).any():
        raise ValueError('Infinite input values are not supported.')
    if clean.isna().all(axis=1).any():
        raise ValueError('A record cannot have every measurement missing.')
    return clean.astype(float)


def native_record_explanation(bundle, clean):
    pipeline = bundle['base_model']
    model = pipeline.named_steps['estimator']
    transformed = pipeline.named_steps['features'].transform(clean)
    terms = np.asarray(model.eval_terms(transformed))
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    reconstructed = expit(intercept + terms.sum(axis=1))
    expected = model.predict_proba(transformed)[:, 1]
    if not np.allclose(reconstructed, expected, atol=1e-9):
        raise ValueError('Native contributions do not reconstruct the base prediction.')
    probability = bundle['model'].predict_proba(clean)[:, 1]
    final_terms, final_intercept = None, None
    if bundle['model'] is pipeline or bundle.get('probability_policy') == 'raw':
        final_terms, final_intercept = terms, intercept
    elif bundle.get('probability_policy') == 'sigmoid':
        calibrated_members = bundle['model'].calibrated_classifiers_
        # A single Platt calibrator applied to the base decision function preserves additivity.
        if len(calibrated_members) == 1 and len(calibrated_members[0].calibrators) == 1:
            response = pipeline.decision_function(clean)
            if np.allclose(response, intercept + terms.sum(axis=1)):
                calibrator = calibrated_members[0].calibrators[0]
                final_terms = -float(calibrator.a_) * terms
                final_intercept = -float(calibrator.a_) * intercept - float(calibrator.b_)
    if final_terms is not None and not np.allclose(expit(final_intercept + final_terms.sum(axis=1)), probability, atol=1e-9):
        raise ValueError('Calibrated contributions do not reconstruct the deployed prediction.')
    return {
        'terms': list(model.term_names_), 'contributions': terms,
        'intercept': intercept, 'base_probability': expected, 'probability': probability,
        'final_contributions': final_terms, 'final_intercept': final_intercept,
        'units': 'base-model log-odds; final contributions include the selected calibration link when exact',
    }


def native_effect(bundle, term_index):
    model = bundle['base_model'].named_steps['estimator']
    term = model.term_features_[term_index]
    if len(term) != 1:
        return None
    feature_index = term[0]
    bins = model.bins_[feature_index][0]
    scores = np.asarray(model.term_scores_[term_index])[1:-1]
    if isinstance(bins, dict):
        labels = list(bins)
        values = [float(model.term_scores_[term_index][bins[label]]) for label in labels]
        return pd.DataFrame({'Value': labels, 'Contribution': values})
    bounds = model.feature_bounds_[feature_index]
    edges = np.r_[bounds[0], bins, bounds[1]]
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({'Value': centers, 'Contribution': scores})
