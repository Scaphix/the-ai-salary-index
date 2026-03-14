from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalMappingEncoder(BaseEstimator, TransformerMixin):
    """Encodes ordinal features using explicit domain-knowledge mappings."""

    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.mappings.items():
            X[col] = X[col].map(mapping)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces categorical values with their frequency
                    (fitted on train only)."""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        self.freq_map_ = {
            col: X[col].value_counts(normalize=True).to_dict()
            for col in self.variables
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].map(self.freq_map_[col]).fillna(0)
        return X
