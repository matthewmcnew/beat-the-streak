import numpy as np


def best_pick_for_each_day_got_hit_percent(clf, X, y):
    df = X.copy()
    df['got_hit'] = y
    df['prob_of_hit'] = np.array(clf.predict_proba(X)).T[1]

    days_with_hit = 0
    for date in df.game_date.unique():

        date_frame = df[df.game_date == date]

        if date_frame.sort_values('prob_of_hit', ascending=False).iloc[0]['got_hit']:
            days_with_hit += 1

    return float(days_with_hit)/len(df.game_date.unique())
