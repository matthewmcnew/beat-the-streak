import pandas as pd
from beat_the_streak.ballparks import hitability_map


def load_dataset_starting_at_day(day='2015-04-30'):
    choices = pd.read_csv('choices.csv')
    choices['player_hash'] = [hash(player) for player in choices.bat_id]
    choices = choices.join(pd.get_dummies(choices.dh_fl, prefix='dh'))
    choices = choices.join(pd.get_dummies(choices.daynight_park_cd, prefix='day_night'))
    choices = choices.join(pd.get_dummies(choices.field_park_cd, prefix='field'))
    choices = choices.join(pd.get_dummies(choices.precip_park_cd, prefix='precip'))
    choices = choices.join(pd.get_dummies(choices.sky_park_cd, prefix='sky'))

    choices = choices[choices['game_date'] > day]
    choices = choices[choices['total_batter_appearances'] > 50]
    choices = choices[choices['total_pitcher_appearances'] > 50]
    choices = choices[choices['bullpen_appearances'] > 30]

    choices['is_high_attendance'] = choices.attend_park_ct.map(lambda attend_park_ct: attend_park_ct > 38000).map(
        {True: 1, False: 0})

    choices['is_coors'] = choices.park_id.map(lambda park: park == 'DEN02').map({True: 1, False: 0})
    choices['is_hity_pitcher'] = choices.pitcher_hitting_average.map(lambda ha: ha > .23).map({True: 1, False: 0})
    choices['is_really_hity_pitcher'] = choices.pitcher_hitting_average.map(lambda ha: ha > .26).map(
        {True: 1, False: 0})

    choices['is_hity_bull'] = choices.bullpen_average.map(lambda ha: ha > .25).map({True: 1, False: 0})

    choices['ballpark'] = choices.park_id.map(hitability_map)
    choices['pitcher_hand_diff'] = choices['hand_pitcher_hitting_average'] - choices['pitcher_hitting_average']
    choices['hitter_hand_diff'] = choices['hand_hitting_average'] - choices['hitting_average']

    return choices[choices.columns[2:]]


not_for_entire_model = [
    'player_hash'
]

not_needed_for_players = [
    'hitting_average'
]

train_cols = [
    'hitting_average',
    'player_hash',
    'bat_home_id',
    'bat_lineup_id',
    'ballpark',
    'dh_T',
    'day_night_N',
    'is_high_attendance',
    'attend_park_ct',

    'field_2',
    'field_4',

    'precip_1',
    'precip_2',
    'precip_4',

    'sky_1',
    'sky_2',
    'sky_3',
    'sky_5',

    'temp_park_ct',
    'wind_speed_park_ct',
    'total_pitcher_appearances',

    'pitcher_hitting_average',
    'is_coors',
    'is_really_hity_pitcher',
    'hitter_hand_diff',
    'pitcher_hand_diff',
    'bullpen_average',
    'is_hity_pitcher',
    'is_hity_bull'
]

necessary_columns = train_cols[:]

necessary_columns.extend([
    'game_date',
    'got_hit',
    'first_name_tx',
    'last_name_tx'])


def test_train_split(day='2015-08-01'):
    choices = load_dataset_starting_at_day()
    choices = choices[necessary_columns].dropna()

    train = choices[choices.game_date < day]
    test = choices[choices.game_date > day]
    return (train, test)


necessary_columns = train_cols[:]

necessary_columns.extend([
    'game_date',
    'got_hit',
    'first_name_tx',
    'last_name_tx'])
