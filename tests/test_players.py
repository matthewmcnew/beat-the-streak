import unittest

import numpy as np

import pandas as pd
from beat_the_streak.players import PlayerModel
from sklearn.base import BaseEstimator, ClassifierMixin

TRAIN_SET = pd.DataFrame({'player_hash': [1, 2], 'something_else': [1, 2], 'got_hit': [0, 1]})
TRAIN_SET_X = TRAIN_SET[['player_hash', 'something_else']]
TRAIN_SET_Y = TRAIN_SET.got_hit


class PlayersModelTest(unittest.TestCase):
    def test_players_model_is_a_sklearn_model(self):
        player_model = PlayerModel()
        assert isinstance(player_model, BaseEstimator)
        assert isinstance(player_model, ClassifierMixin)

    def test_creates_new_classifer_for_each_player_hash(self):

        player_model = PlayerModel(cls=OtherModel)

        player_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        assert OtherModel.NUMBER_OF_INSTANCES == 2

    def test_fits_each_classifer_with_unique_values(self):

        player_model = PlayerModel(cls=OtherModel)
        player_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        assert OtherModel.PLAYER_1.fitted_x.equals(TRAIN_SET_X[TRAIN_SET_X['player_hash'] == 1])
        assert OtherModel.PLAYER_1.fitted_y.equals(TRAIN_SET[TRAIN_SET['player_hash'] == 1].got_hit)

        assert OtherModel.PLAYER_2.fitted_x.equals(TRAIN_SET_X[TRAIN_SET_X['player_hash'] == 2])
        assert OtherModel.PLAYER_2.fitted_y.equals(TRAIN_SET[TRAIN_SET['player_hash'] == 2].got_hit)

    def test_predicts_probability_from_players_classifier(self):

        players_model = PlayerModel(cls=OtherModel)
        players_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        probs = players_model.predict_proba(TRAIN_SET_X)

        assert np.allclose(probs, [[.4, .8], [.8, .4]])

    def test_predict_uses_players_model_prob(self):

        players_model = PlayerModel(cls=OtherModel)
        players_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        predict = players_model.predict(TRAIN_SET_X)

        assert np.allclose(predict, [1, 0])


class OtherModel(BaseEstimator, ClassifierMixin):
    NUMBER_OF_INSTANCES = 0
    PLAYER_1 = None
    PLAYER_2 = None

    def __init__(self):
        OtherModel.NUMBER_OF_INSTANCES += 1

        if OtherModel.PLAYER_1 is not None:
            OtherModel.PLAYER_2 = self
        else:
            OtherModel.PLAYER_1 = self

    def fit(self, X, y):
        self.fitted_x = X
        self.fitted_y = y

    def predict_proba(self, X):
        if self is OtherModel.PLAYER_1:
            return np.array([.4, .8])
        elif self is OtherModel.PLAYER_2:
            return np.array([.8, .4])
