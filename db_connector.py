import pymysql.cursors
import numpy as np

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
            away_score = batsmen_model(player['away_matches'],player['away_innings'],player['away_average'],player['away_100s'],player['away_50s'])
            player_list.append([player['id'],career_score,player['form_average'],away_score])
        
        np_arr = np.array(player_list)
        max_career = np.max(np_arr[:,1])
        max_recent = np.max(np_arr[:,2])
        print(max_career)
        print(max_recent)
        for player in np_arr:
            career_score = player[1]
            recent_score = player[2]
            normalized_career_score = max_career/career_score
            normalized_recent_score = max_recent/recent_score
            
            player[1] = normalized_career_score
            player[2] = normalized_recent_score
        
        print(np_arr)
finally:
    connection.close()
