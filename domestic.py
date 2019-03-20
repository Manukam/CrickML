import pymysql.cursors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='crickml',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

def batsmen_performance_model(matches, innings, average, hundreds, fifties):
    if(innings <= 0):
        return 0.0
    else:
        u = innings/matches
        v = (20 * hundreds) + (5 * fifties)
        w = (0.3 * v) + (0.7 * average)
        return u * w

def batsmen_model(matches, innings, average, hundreds, fifties):
    if(innings <= 0):
        return 0.0
    else:
        u = innings/matches
        v = (20 * hundreds) + (5 * fifties)
        w = (0.3 * v) + (0.7 * average)
        return u, v * w

try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `domestic_stats`"
        cursor.execute(sql)
        result = cursor.fetchall()
        player_list = []
        for player in result:
            career_scores = batsmen_model(player['overall_matches'], player['overall_innings'],
                                         player['overall_average'], player['overall_100s'], player['overall_50s'])
            player_list.append(
                career_scores)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `domestic_stats`"
        cursor.execute(sql)
        result = cursor.fetchall()
        intl_performance_list = []
        performance_list = []
        for player in result:
            performance_score = batsmen_performance_model(player['intl_matches'], player['intl_innings'],
                                         player['intl_average'], player['intl_100s'], player['intl_50s'])
                                         
            intl_performance_list.append(
                [performance_score])
    
    np_intl_performances_list = np.array(intl_performance_list)
    mean_performance = sum(np_intl_performances_list[:,0])/len(np_intl_performances_list)

    for performance in intl_performance_list:
            if(performance <= mean_performance):
                performance_list.append(0)
            else:
                performance_list.append(1)


finally:
    connection.close()
    # print(np_players)



np_players = np.array(player_list)
np_performances = np.array(performance_list)

# print(np_players)
max_batting_pos = np.max(np_players[:, 0])
max_milestone_score = np.max(np_players[:, 1])
# max_runs_score = np.max(np_players[:, 2])
# max_performance = np.max(np_performances[:, 0])

# print(np_players)
# print(np_performances)

for player in np_players:
    batting_pos_score = player[0]
    batting_milestone_score = player[1]
    # batting_runs_score = player[2]

    normalized_batting_pos_score = batting_pos_score/max_batting_pos
    normalized_batting_milestone_score = batting_milestone_score/max_milestone_score
    # normalized_runs_score = batting_runs_score/max_runs_score

    player[0] = normalized_batting_pos_score
    player[1] = normalized_batting_milestone_score
    # player[2] = normalized_runs_score
    
# for player in np_performances:
#     performance_score = player[0]
#     normalized_performance_score = performance_score/max_performance
#     player[0] = normalized_performance_score


feature_train, feature_test, target_train, target_test = train_test_split(
    np_players, np_performances, test_size=0.20, random_state=42)

# print(feature_train)
# print(target_train)

svm_clf = SVC(C=1000, kernel='rbf', gamma=0.001, probability=True)
svm_clf.fit(feature_train, target_train)
svm_pred = svm_clf.predict(feature_test)
svm_pred_prob = svm_clf.predict_proba(feature_test)
print(svm_pred_prob)
acc = accuracy_score(svm_pred, target_test)
print('Accuracy :', acc)
print(classification_report(target_test, svm_pred))

feature1_for_plot = np_players[:,0]
feature2_for_plot = np_players[:,1]

for feature1, feature2, target in np.nditer([feature1_for_plot, feature2_for_plot, np_performances]):
    if(target == 1):
        plt.scatter(feature1, feature2, color='r')
    else:
        plt.scatter(feature1, feature2, color='b')
    #  plt.scatter(feature1, feature2, color='b')

plt.scatter(feature1_for_plot[0],
            np_performances[0], color='b', label="Below")
plt.scatter(feature1_for_plot[0],
            np_performances[0], color='r', label="Above")
# plt.xlabel(X.columns[0], size=14)
# plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)

plt.xlabel('Domestic Performance')
plt.ylabel('INTL')
plt.legend()
plt.show()
