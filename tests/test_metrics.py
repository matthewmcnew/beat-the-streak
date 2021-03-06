import unittest

import pandas as pd
from beat_the_streak.metrics import BestPickForEachDayGotHitPercent

CHOICES = pd.DataFrame({
    'game_date': ['12-12-10', '12-12-10', '12-12-11', '12-12-11'],
    'some_feature': [1, 2, 3, 4],
    'is_first_in_group': [1, 0, 1, 0]},
    index=['a', 'b', 'c', 'd'])
X = CHOICES[['some_feature', 'is_first_in_group']]


class BestPickForEachDayGotHitPercentTest(unittest.TestCase):
    def test_returns_the_percent_of_days_where_models_highest_prob_got_a_hit(self):
        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                assert x_to_predict is X

                return [[.1, .9], [.8, .1], [.9, .1], [.8, .2]]

        y = pd.Series([1, 0, 0, 0])

        score = BestPickForEachDayGotHitPercent(CHOICES.game_date)(StubClassifier(), X, y)

        assert score == .5

    def test_handles_when_not_highest_prob_got_hit(self):
        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                assert x_to_predict is X

                return [[.8, .1], [.2, .8], [.9, .1], [.8, .2]]

        y = pd.Series([1, 0, 1, 0])

        score = BestPickForEachDayGotHitPercent(CHOICES.game_date)(StubClassifier(), X, y)

        assert score == 0

    def test_uses_index_to_match_dates(self):
        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                return [[.8, .1], [.2, .8]]

        y = pd.Series([1, 0])

        score = BestPickForEachDayGotHitPercent(CHOICES.game_date)(StubClassifier(), X[X.is_first_in_group == 1], y)

        assert score == .5

    def test_allows_number_of_choices_per_day(self):
        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                return [[.8, .1], [.2, .8], [.9, .1], [.8, .2]]

        y = pd.Series([1, 1, 1, 0])

        score = BestPickForEachDayGotHitPercent(CHOICES.game_date, number_of_choices_per_day=2)(StubClassifier(), X, y)

        assert score == .75
