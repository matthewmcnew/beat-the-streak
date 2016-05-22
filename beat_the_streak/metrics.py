import numpy as np
import pandas as pd


class BestPickForEachDayGotHitPercent(object):
    def __init__(self, game_dates):
        self.game_dates = game_dates

    def __call__(self, clf, X, y, *args, **kwargs):
        df = X.copy()
        df['prob_of_hit'] = np.array(clf.predict_proba(X)).T[1]
        df['got_hit'] = [got_hit for got_hit in y]
        # noinspection PyTypeChecker
        df = pd.concat([df, self.game_dates], join='inner', axis=1)

        days_with_hit = 0
        for date in df.game_date.unique():
            date_frame = df[df.game_date == date]

            if date_frame.sort_values('prob_of_hit', ascending=False).iloc[0]['got_hit']:
                days_with_hit += 1
        print len(df.game_date.unique())

        return float(days_with_hit) / len(df.game_date.unique())
