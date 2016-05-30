def play(clf, data):
    your_streak = 0

    for game_date in data.sort_values('game_date').game_date.unique():
        print '   '
        print "Your current streak: " + str(your_streak)

        narrowed_choices = data[data.game_date == game_date].copy()

        narrowed_choices['prob'] = clf.predict_proba(narrowed_choices).T[0]
        narrowed_choices['name'] = narrowed_choices['first_name_tx'] + ' ' + narrowed_choices['last_name_tx']
        narrowed_choices = narrowed_choices.sort_values('prob')

        print 'Game Date to choose ' + game_date
        print narrowed_choices[['hitting_average', 'pitcher_hitting_average', 'name', 'prob']].tail(10)
        print ''
        print 'Make your choice:'

        who = raw_input()

        choice = narrowed_choices.loc[int(who)]
        got_hit = choice.got_hit
        if got_hit == 1:
            print 'hooray! ' + choice['first_name_tx'] +  ' ' + choice['last_name_tx'] + ' got a hit!'
            your_streak += 1
        else:
            print 'No hit! Better Luck Next time!'
            return
