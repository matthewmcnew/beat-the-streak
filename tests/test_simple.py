import unittest

import numpy as np

import pandas as pd
from beat_the_streak.simple import SimpleHittingModel
from sklearn.base import BaseEstimator, ClassifierMixin

TRAIN_SET = pd.DataFrame({'hitting_average': [.5, 4], 'got_hit': [0, 1]})
TRAIN_SET_X = TRAIN_SET[['hitting_average']]
TRAIN_SET_Y = TRAIN_SET.got_hit

class SimpleHittingModelTest(unittest.TestCase):
   def test_is_scikit_learn_module(self):
        simple_model = SimpleHittingModel()
        assert isinstance(simple_model, BaseEstimator)
        assert isinstance(simple_model, ClassifierMixin)

   def test_predict_uses_hitting_average(self):
        simple_model = SimpleHittingModel(average_number_of_attempts=1)

        simple_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        test = pd.DataFrame({'hitting_average': [.8, .4, .6]})
        predict = simple_model.predict(test)

        assert predict == [1, 0, 1]

   def test_predict_proba_uses_hitting_average(self):
        simple_model = SimpleHittingModel(average_number_of_attempts=1)

        simple_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        test = pd.DataFrame({'hitting_average': [.8, .4, .6]})

        predict_prob = simple_model.predict_proba(test)
        assert np.allclose(predict_prob, np.array([[.2, .8], [.6, .4], [.4, .6]]))


   def test_predict_prob_a_uses_hitting_average_and_average_number_of_attempts(self):
        simple_model = SimpleHittingModel(average_number_of_attempts=2)

        simple_model.fit(TRAIN_SET_X, TRAIN_SET_Y)

        test = pd.DataFrame({'hitting_average': [.8]})

        predict_prob = simple_model.predict_proba(test)

        prob_of_no_hit = (.2) * (.2)
        prob_of_hit = 1 - prob_of_no_hit
        assert np.allclose(predict_prob, [[prob_of_no_hit, prob_of_hit]])
