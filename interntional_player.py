from player import Player


class International_Player(Player):
    def __init__(self, id, name, overall_matches, overall_innings, overall_runs, overall_average, overall_strike_rate, overall_100s, overall_50s, home_matches, home_innings, home_runs,
                 home_average, home_strike_rate, home_100s, home_50s, away_matches, away_innings, away_runs, away_average, away_strike_rate, away_100s, away_50s,
                 form_matches, form_innings, form_runs, form_average, form_strike_rate, form_100s, form_50s):
        Player.__init__(self, id, name, overall_matches, overall_innings, overall_runs,
                        overall_average, overall_strike_rate, overall_100s, overall_50s)
        self.home_matches = home_matches
        self.home_innings = home_innings
        self.home_runs = home_runs
        self.home_average = home_average
        self.home_strike_rate = home_strike_rate
        self.home_100s = home_100s
        self.home_50s = home_50s
        self.away_matches = away_matches
        self.away_innings = away_innings
        self.away_runs = away_runs
        self.away_average = away_average
        self.away_strike_rate = away_strike_rate
        self.away_100s = away_100s
        self.away_50s = away_50s
        self.form_matches = form_matches
        self.form_innings = form_innings
        self.form_runs = form_runs
        self.form_average = form_average
        self.form_strike_rate = form_strike_rate
        self.form_100s = form_100s
        self.form_50s = form_50s

    def calculate_home_score(self, matches, innings, runs, average, strike_rate, hundreds, fifties):
        if(innings <= 0):
            return 0.0
        else:
            u = innings/matches
            v = (20 * hundreds) + (5 * fifties)
            w = (0.3 * v) + (0.7 * average)
            return u * w

    def calculate_away_score(self, matches, innings, runs, average, strike_rate, hundreds, fifties):
        if(innings <= 0):
            return 0.0
        else:
            u = innings/matches
            v = (20 * hundreds) + (5 * fifties)
            w = (0.3 * v) + (0.7 * average)
            return u * w

    def calculate_recent_scrore(self, matches, innings, runs, average, strike_rate, hundreds, fifties):
        if(innings <= 0):
            return 0.0
        else:
            u = innings/matches
            v = (20 * hundreds) + (5 * fifties)
            w = (0.3 * v) + (0.7 * average)
            return u * w
