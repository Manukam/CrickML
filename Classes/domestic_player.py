from Classes.player import Player


class Domestic_Player(Player):
    def __init__(self, id, name, overall_matches, overall_innings, overall_runs, overall_average, overall_strike_rate, overall_100s, overall_50s
                 ):
        Player.__init__(self, id, name, overall_matches, overall_innings, overall_runs,
                        overall_average, overall_strike_rate, overall_100s, overall_50s)

    def calculate_domestic_score(self):
        if(self.overall_innigs <= 0):
            return 0.0
        else:
            u = self.overall_innigs/self.overall_matches
            v = (20 * self.overall_100s) + (5 * self.overall_50s)
            w = (0.3 * v) + (0.7 * self.overall_average)
            return u * w
