class FeatureSelector(object):
    def __init__(self, features = []):
        self.features = features

    def transform(self, X, y):
        return X[self.features]

    def fit(self, X, y):
        return self
