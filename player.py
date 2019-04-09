class Player:
    def __init__(self, id, name, overall_matches, overall_innings, overall_runs, overall_average, overall_strike_rate, overall_100s, overall_50s):
        self.id = id
        self.name = name
        self.overall_matches = overall_matches
        self.overall_innigs = overall_innings
        self.overall_runs = overall_runs
        self.overall_average = overall_average
        self.overall_strike_rate = overall_strike_rate
        self.overall_100s = overall_100s
        self.overall_50s = overall_50s

    def calculate_overall_score(self, matches, innings, hundreds, fifties, average):
        if(innings <= 0):
            return 0.0
        else:
            u = innings/matches
            v = (20 * hundreds) + (5 * fifties)
            w = (0.3 * v) + (0.7 * average)
            return u * w
