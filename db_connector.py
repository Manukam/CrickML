import pymysql.cursors
import numpy as np
from sklearn.cross_validation import train_test_split

def batsmen_model(matches, innings, average, hundreds, fifties):
    u = innings/matches
    v = (20 * hundreds) + (5 * fifties)
    w = (0.3 * v) + (0.7 * average)
    return u * w

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='crickml',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `pre_wc_2011`"
        cursor.execute(sql)
        result = cursor.fetchall()
        player_list = []
        for player in result:
            career_score = batsmen_model(player['overall_matches'],player['overall_innings'],player['overall_average']
            ,player['overall_100s'],player['overall_50s'])

            away_score = batsmen_model(player['away_matches'],player['away_innings'],
            player['away_average'],player['away_100s'],player['away_50s'])

            home_score =  batsmen_model(player['home_matches'],player['home_innings'],
            player['home_average'],player['home_100s'],player['home_50s'])

            player_list.append([player['id'],career_score,player['form_average'],away_score,home_score])
        
        np_arr = np.array(player_list)
        max_career = np.max(np_arr[:,1])
        max_recent = np.max(np_arr[:,2])
        max_away = np.max(np_arr[:,3])
        max_home = np.max(np_arr[:,4])
        print(max_career)
        print(max_recent)
        for player in np_arr:
            career_score = player[1]
            recent_score = player[2]
            away_score = player[3]
            home_score = player[4]

            normalized_career_score = max_career/career_score
            normalized_recent_score = max_recent/recent_score
            normalized_away_score = max_away/away_score
            normalized_home_score = max_home/home_score
            
            player[1] = normalized_career_score
            player[2] = normalized_recent_score
            player[3] = normalized_away_score
            player[4] = normalized_home_score
        
        print(np_arr)

        a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)
finally:
    connection.close()
