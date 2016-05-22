import unittest

import pandas as pd
from beat_the_streak.metrics import best_pick_for_each_day_got_hit_percent


class BestPickForEachDayGotHitPercentTest(unittest.TestCase):

    def test_returns_the_percent_of_days_where_models_highest_prob_got_a_hit(self):

        X = pd.DataFrame({'game_date': ['12-12-10', '12-12-10', '12-12-11', '12-12-11']})

        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                assert x_to_predict is X

                return [[.1, .9], [.8, .1], [.9, .1], [.8, .2]]

        y = pd.Series([1, 0, 0, 0])

        score = best_pick_for_each_day_got_hit_percent(StubClassifier(), X, y)

        assert score == .5

    def test_handles_when_not_highest_prob_got_hit(self):

        X = pd.DataFrame({'game_date': ['12-12-10', '12-12-10', '12-12-11', '12-12-11']})

        class StubClassifier(object):
            def predict_proba(self, x_to_predict):
                assert x_to_predict is X

                return [[.8, .1], [.2, .8], [.9, .1], [.8, .2]]

        y = pd.Series([1, 0, 1, 0])

        score = best_pick_for_each_day_got_hit_percent(StubClassifier(), X, y)

        assert score == 0

