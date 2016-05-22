from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleHittingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, average_number_of_attempts=5):
        self.average_number_of_attempts = average_number_of_attempts

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [self._predict_for_x(x[1]) for x in X.iterrows()]

    def predict_proba(self, X):
        return [self._prob_number_attempts(x[1]) for x in X.iterrows()]

    def _predict_for_x(self, x):
        if self._prob_number_attempts(x) > .5:
            return 1
        return 0

    def _prob_number_attempts(self, x):
        return 1 - (1 - x['hitting_average'])**self.average_number_of_attempts
