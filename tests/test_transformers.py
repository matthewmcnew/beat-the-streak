import unittest

import pandas as pd
from beat_the_streak.transformers import FeatureSelector

TRAIN_SET = pd.DataFrame({'hitting_average': [.5, 4], 'useless_average': [90, 91], 'got_hit': [0, 1]})
TRAIN_SET_X = TRAIN_SET[['hitting_average', 'useless_average']]
TRAIN_SET_Y = TRAIN_SET.got_hit

class FeatureSelectorTest(unittest.TestCase):

    def test_is_scikit_transformer(self):
        selector = FeatureSelector().fit(TRAIN_SET_X, TRAIN_SET_Y)

        assert isinstance(selector, FeatureSelector)

    def test_selects_features_in_a_pipeline(self):
        selector = FeatureSelector(['hitting_average'])
        transform = selector.transform(TRAIN_SET_X, TRAIN_SET_Y)

        assert transform.equals(TRAIN_SET_X[['hitting_average']])




