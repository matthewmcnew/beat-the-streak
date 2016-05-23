import numpy as np
import pandas as pd


class BestPickForEachDayGotHitPercent(object):
    def __init__(self, game_dates, number_of_choices_per_day=1):
        self.number_of_choices_per_day = number_of_choices_per_day
        self.game_dates = game_dates

    def __call__(self, clf, X, y, *args, **kwargs):
        df = X.copy()
        df['prob_of_hit'] = np.array(clf.predict_proba(X)).T[1]
        df['got_hit'] = [got_hit for got_hit in y]
        # noinspection PyTypeChecker
        df = pd.concat([df, self.game_dates], join='inner', axis=1)

        days_with_hit = 0
        for date in df.game_date.unique():

            sorted_prediction_for_day = df[df.game_date == date].sort_values('prob_of_hit', ascending=False)

            for index in range(self.number_of_choices_per_day):
                if sorted_prediction_for_day.iloc[index]['got_hit']:
                    days_with_hit += 1

        return float(days_with_hit) / (len(df.game_date.unique()) * self.number_of_choices_per_day)
