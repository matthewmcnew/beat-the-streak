import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class PlayerModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_cls=LogisticRegression):
        self.model_cls = model_cls

    def fit(self, X, y):
        pandasy = pd.Series(y, index=X.index)

        self.player_models_ = dict()
        for player_hash in X.player_hash.unique():

            # untested
            if len(pandasy[X.player_hash == player_hash].unique()) != 2:
                pass
            else:
                player_model = self.model_cls()
                player_model.fit(X[X.player_hash == player_hash], pandasy[X.player_hash == player_hash])

                self.player_models_[player_hash] = player_model
        return self

    def predict(self, X):
        return [0 if prob[0] > .5 else 1 for prob in self.predict_proba(X)]

    def predict_proba(self, X_to_predict):
        probs = []

        for _, row in X_to_predict.iterrows():

            # print row.player_hash
            # print row.player_hash in self.player_models_.keys()
            index = -1

            if row.player_hash in self.player_models_.keys():
                index = self.player_models_.keys().index(row.player_hash)

            if index != -1:
                key = self.player_models_.keys()[index]
            else:
                key = -1

            # untested
            player_model = self.player_models_.get(key, DummyModel())

            prob = player_model.predict_proba(row.reshape(1, -1))[0]
            probs.append(prob)

        return np.array(probs)


class DummyModel(object):
    def predict_proba(self, X):
        return [[1, 0]]
