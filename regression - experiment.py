import pymysql.cursors
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

# np.set_printoptions(suppress=True)

def batsmen_model(matches, innings, average, hundreds, fifties):
    if(innings <= 0):
        return 0.0
    else:
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
            career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                         player['overall_average'], player['overall_100s'], player['overall_50s'])

            away_score = batsmen_model(player['away_matches'], player['away_innings'],
                                       player['away_average'], player['away_100s'], player['away_50s'])

            home_score = batsmen_model(player['home_matches'], player['home_innings'],
                                       player['home_average'], player['home_100s'], player['home_50s'])

            # condition_performances = away_score + home_score

            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append([career_score,away_score,home_score])

    with connection.cursor() as cursor:
        sql = "SELECT * FROM `pre_wc_2015`"
        cursor.execute(sql)
        result = cursor.fetchall()
        # player_list = []
        for player in result:
            career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                         player['overall_average'], player['overall_100s'], player['overall_50s'])

            away_score = batsmen_model(player['away_matches'], player['away_innings'],
                                       player['away_average'], player['away_100s'], player['away_50s'])

            home_score = batsmen_model(player['home_matches'], player['home_innings'],
                                       player['home_average'], player['home_100s'], player['home_50s'])

            # condition_performances = away_score + home_score

            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append([career_score,away_score,home_score])

    # with connection.cursor() as cursor:
    #     sql = "SELECT * FROM `pre_ct_2017`"
    #     cursor.execute(sql)
    #     result = cursor.fetchall()
    #     # player_list = []
    #     for player in result:
    #         career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
    #                                      player['overall_average'], player['overall_100s'], player['overall_50s'])

    #         away_score = batsmen_model(player['away_matches'], player['away_innings'],
    #                                    player['away_average'], player['away_100s'], player['away_50s'])

    #         home_score = batsmen_model(player['home_matches'], player['home_innings'],
    #                                    player['home_average'], player['home_100s'], player['home_50s'])

    #         # condition_performances = away_score + home_score

    #         # career_score = career_score + condition_performances

    #         # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

    #         player_list.append([career_score,away_score,home_score])

    with connection.cursor() as cursor:
        sql = "SELECT * FROM `pre_ct_2013`"
        cursor.execute(sql)
        result = cursor.fetchall()
        # player_list = []
        for player in result:
            career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                         player['overall_average'], player['overall_100s'], player['overall_50s'])

            away_score = batsmen_model(player['away_matches'], player['away_innings'],
                                       player['away_average'], player['away_100s'], player['away_50s'])

            home_score = batsmen_model(player['home_matches'], player['home_innings'],
                                       player['home_average'], player['home_100s'], player['home_50s'])

            # condition_performances = away_score + home_score

            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append([career_score,away_score,home_score])

        np_players = np.array(player_list)
        # print(np_players) 
        # print(np_players.shape)
        # exit()
        max_career = np.max(np_players[:, 0])
        # max_recent = np.max(np_players[:, 1])
        max_away = np.max(np_players[:, 1])
        max_home = np.max(np_players[:, 2])
        print(max_career)
        print(max_away)
        print(max_home)
        # exit()
        for player in np_players:
            career_score = player[0]
            # recent_score = player[1]
            away_score = player[1]
            home_score = player[2]

            normalized_career_score = career_score/max_career
            player[0] = normalized_career_score

            # if(recent_score != 0):
            #     normalized_recent_score = recent_score/max_recent
            #     player[1] = normalized_recent_score

            if(away_score != 0):
                normalized_away_score = away_score/max_away
                player[1] = normalized_away_score

            if(home_score != 0):
                normalized_home_score = home_score/max_home
                player[2] = normalized_home_score

    # a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `wc_2011`"
        cursor.execute(sql)
        result = cursor.fetchall()
        performance_list = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            performance_list.append([tournement_score])

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `wc_2015`"
        cursor.execute(sql)
        result = cursor.fetchall()
        # performance_list = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            performance_list.append([tournement_score])

    # with connection.cursor() as cursor:
    #     # Read a single record
    #     sql = "SELECT * FROM `ct_2017`"
    #     cursor.execute(sql)
    #     result = cursor.fetchall()
    #     # performance_list = []
    #     for player in result:
    #         tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
    #                                          player['overall_average'], player['overall_100s'], player['overall_50s'])
    #         performance_list.append([tournement_score])

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `ct_2013`"
        cursor.execute(sql)
        result = cursor.fetchall()
        # performance_list = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            performance_list.append([tournement_score])

        
    np_performances = np.array(performance_list)
    max_tournament_score = np.max(np_performances[:, 0])

    # itemindex = np.where(np_players[:,0]==1)
    # print(itemindex)
    # print(np_players)
    # exit()

    # exit()

    for player in np_performances:
        tourney_score = player[0]

        normalized_tournement_score = tourney_score/max_tournament_score

        # print(normalized_tournement_score)
        player[0] = normalized_tournement_score

finally:
    connection.close()
    # print(np_performances)
    # print(np_players)
    # exit()
# print(np_players)
feature_train, feature_test, target_train, target_test = train_test_split(
    np_players, np_performances, test_size=0.20, random_state=42)
# print(feature_train)
train_color = "b"
test_color = "r"


reg = linear_model.Ridge(alpha=.5)
reg.fit(feature_train, target_train)
# print(reg.coef_)
# print(reg.intercept_)
pree = reg.predict(feature_test)
print(reg.score(feature_train,target_train))
# print(target_test)
# print(pree)

# draw the scatterplot, with color-coded training and testing points
feature_test_for_plot = feature_test[:, 0]
feature_train_for_plot = feature_train[:, 0]

target_test_for_plot = target_test[:,0]
target_train_for_plot = target_train[:,0]

print(feature_train_for_plot)
print(feature_test_for_plot)
# print(target_train)
# exit()
for feature, target in np.nditer([feature_test_for_plot, target_test_for_plot]):
    plt.scatter(feature, target, color=test_color)
for feature, target in np.nditer([feature_train_for_plot, target_train_for_plot]):
    plt.scatter(feature, target, color=train_color)

# print(target_test[0])
# labels for the legend
# plt.scatter(feature_test_for_plot[0],
#             target_test_for_plot[0], color=test_color, label="test")
# plt.scatter(feature_test_for_plot[0],
#             target_test_for_plot[0], color=train_color, label="train")


# draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test))
    # print('name error')
except NameError:
    pass

# reg.fit(feature_test, target_test)
# print(reg.coef_)
# plt.plot(feature_train, reg.predict(feature_train), color="b")
plt.xlabel('Career Performance')
plt.ylabel('Tournament Score')
plt.legend()
plt.show()
