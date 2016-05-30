from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        if not features:
            features = []
        self.features = features

    # noinspection PyUnusedLocal
    def transform(self, X, y=None):
        return X[self.features]

    def fit(self, X, y):
        return self
