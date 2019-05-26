import pymysql.cursors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import math as math
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from Classes.domestic_player import Domestic_Player


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
        return v

def domestic_model_initialise():
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='',
                                db='crickml',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT * FROM `domestic_stats`"
            cursor.execute(sql)
            result = cursor.fetchall()
            player_list = []
            for player in result:
                career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                            player['overall_average'], player['overall_100s'], player['overall_50s'])
                player_list.append(
                    [player['overall_average'] * player['overall_strike_rate'],career_score])

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
    finally:
        print('done')
        # return pre_performance


    np_intl_performances_list = np.array(intl_performance_list)
    mean_performance = sum(np_intl_performances_list[:,0])/len(np_intl_performances_list)

    for performance in intl_performance_list:
        if(performance <= mean_performance):
            performance_list.append(0)
        else:
            performance_list.append(1)


    # finally:
    #     connection.close()
    # # print(np_players)
    np_players = np.array(player_list)
    np_players = np_players.astype(float)
    np_performances = np.array(performance_list)

    max_batting_pos = np.max(np_players[:, 0])
    max_milestone_score = np.max(np_players[:, 1])

    for player in np_players:
        batting_pos_score = player[0]
        batting_milestone_score = player[1]
        # batting_runs_score = player[2]
        # print(batting_pos_score)
        # print(max_batting_pos)

    normalized_batting_pos_score = batting_pos_score/max_batting_pos
    normalized_batting_milestone_score = batting_milestone_score/max_milestone_score
    # normalized_runs_score = batting_runs_score/max_runs_score

    # print(normalized_batting_pos_score)
    # exit()
    player[0] = normalized_batting_pos_score
    player[1] = normalized_batting_milestone_score
    # player[2] = normalized_runs_score

    sm = SMOTE(random_state=42)
    np_players_resampled, np_performances_resampled = sm.fit_resample(np_players, np_performances)

    feature_train, feature_test, target_train, target_test = train_test_split(
    np_players_resampled, np_performances_resampled, test_size=0.20, random_state=42)

    print("Training Domestoc Models")
    print(feature_test)
    svm_clf = SVC(C=1000, kernel='sigmoid', gamma=0.001, probability=True)
    svm_clf.fit(feature_train, target_train)
    svm_pred = svm_clf.predict(feature_test)
    svm_pred_prob = svm_clf.predict_proba(feature_test)
    # print(svm_pred_prob)
    # acc = accuracy_score(svm_pred, target_test)
    # print('Accuracy :', acc)
    # print(classification_report(target_test, svm_pred))

    gnb = GaussianNB()
    gnb.fit(feature_train, target_train)
    nb_pred_prob = gnb.predict_proba(feature_test)
    nb_pred = gnb.predict(feature_test)
    # acc = accuracy_score(nb_pred, target_test)
    # print('Accuracy :', acc)
    # print(classification_report(target_test, nb_pred))

    desT = DecisionTreeClassifier()
    desT.fit(feature_train, target_train)
    desc_pred = desT.predict(feature_test)
    desc_pred_prob = desT.predict_proba(feature_test)


    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp_clf.fit(feature_train, target_train)
    mlp_pred_prob = mlp_clf.predict_proba(feature_test)
    mlp_pred = mlp_clf.predict(feature_test)

# acc = accuracy_score(desc_pred, target_test)
# print('Accuracy :', acc)
# print(classification_report(target_test, desc_pred))

    miss_nb = 0
    for index, pred in  enumerate(nb_pred):
        if(pred != target_test[index]):
            miss_nb += 1
    amt_say_nb = 1/2 * (math.log((1-(miss_nb/119))/(miss_nb/119))) 

    miss_mlp = 0
    for index, pred in  enumerate(mlp_pred):
        if(pred != target_test[index]):
            miss_mlp += 1

    amt_say_mlp = 1/2 * (math.log((1-(miss_mlp/119))/(miss_mlp/119)))

    miss_svm = 0
    for index, pred in  enumerate(svm_pred):
        if(pred != target_test[index]):
            miss_svm += 1

    amt_say_svm = 1/2 * (math.log((1-(miss_svm/119))/(miss_svm/119)))

    miss_desc = 0
    for index, pred in  enumerate(desc_pred):
        if(pred != target_test[index]):
            miss_desc += 1

    amt_say_desc = 1/2 * (math.log((1-(miss_desc/119))/(miss_desc/119)))

    print('Amount of say NB :', amt_say_nb)
    print('Amount of say MLP :', amt_say_mlp)
    print('Amount of say SVM :', amt_say_svm)
    print('Amount of say Descision Tree :', amt_say_desc)

    return connection, gnb, mlp_clf, svm_clf, desT, amt_say_desc, amt_say_mlp, amt_say_nb, amt_say_svm, max_batting_pos, max_milestone_score, feature_train, feature_test, target_train, target_test

connection, nb, mlp, svm, dest, desc_say, mlp_say, nb_say, svm_say, max_batting_pos, max_milestone_score, feature_train, feature_test, target_train, target_test = domestic_model_initialise()

def build_domestic_player(player):
    print("Building player...")
    dom_player = Domestic_Player(player["id"], player['player_name'], player['overall_matches'], player['overall_innings'], player['overall_runs'], player['overall_average'],
                                     player['overall_strike_rate'], player['overall_100s'], player['overall_50s'])
    return dom_player

def scale_domestic_features(domestic_features):
    normalized_batting_pos_score = domestic_features[0]/max_batting_pos
    normalized_batting_milestone_score = domestic_features[1]/max_milestone_score
    return [normalized_batting_milestone_score, normalized_batting_pos_score]

def build_domestic_features(player):
    domestic_score = player.calculate_domestic_score()
    pos_score = player.overall_strike_rate * player.overall_average
    return [ pos_score,domestic_score ]

def get_domestic_predictions(domestic_player):
    print(domestic_player)
    nb_pred_prob = nb.predict_proba(domestic_player)
    # nb_pred = nb.predict(player_list)
    mlp_pred_prob = mlp.predict_proba(domestic_player)
    # mlp_pred = mlp.predict(player_list)
    # svm_pred = svm.predict(player_list)
    svm_pred_prob = svm.predict_proba(domestic_player)
    # desc_pred = dest.predict(player_list)
    desc_pred_prob = dest.predict_proba(domestic_player)

    return nb_pred_prob, mlp_pred_prob, svm_pred_prob, desc_pred_prob
# final_predictions = []

# for index, initial_nb_pred in enumerate(nb_pred_prob):
#     weighted_nb_prediction0 = amt_say_nb * (nb_pred_prob[index][0])
#     weighted_nb_prediction1 = amt_say_nb * (nb_pred_prob[index][1])

#     weighted_mlp_prediction0 = amt_say_mlp * (mlp_pred_prob[index][0])
#     weighted_mlp_prediction1 = amt_say_mlp * (mlp_pred_prob[index][1])

#     weighted_svm_prediction0 = amt_say_svm * (svm_pred_prob[index][0])
#     weighted_svm_prediction1 = amt_say_svm * (svm_pred_prob[index][1])

#     weighted_desc_prediction0 = amt_say_desc * (desc_pred_prob[index][0])
#     weighted_desc_prediction1 = amt_say_desc * (desc_pred_prob[index][1])

#     mean_weighted_prediction0 = (weighted_mlp_prediction0 + weighted_nb_prediction0 + weighted_svm_prediction0 + weighted_desc_prediction0 ) / 4
#     mean_weighted_prediction1 = (weighted_mlp_prediction1 + weighted_nb_prediction1 + weighted_svm_prediction1 + weighted_desc_prediction1) / 4

#     # print('Mean Weighted 0 :', mean_weighted_prediction0)
#     # print('Mean Weighted 1 :', mean_weighted_prediction1)
#     if(mean_weighted_prediction0 > mean_weighted_prediction1):
#         final_predictions.append(0)
#     else:
#         final_predictions.append(1)

# acc = accuracy_score(final_predictions, target_test)
# print('Accuracy :', acc)
# print(classification_report(target_test, final_predictions))


# feature1_for_plot = np_players[:,0]
# feature2_for_plot = np_players[:,1]

# for feature1, feature2, target in np.nditer([feature1_for_plot, feature2_for_plot, np_performances]):
#     if(target == 1):
#         plt.scatter(feature1, feature2, color='r')
#     else:
#         plt.scatter(feature1, feature2, color='b')
#     #  plt.scatter(feature1, feature2, color='b')

# plt.scatter(feature1_for_plot[0],
#             np_performances[0], color='b', label="Below")
# plt.scatter(feature1_for_plot[0],
#             np_performances[0], color='r', label="Above")
# # plt.xlabel(X.columns[0], size=14)
# # plt.ylabel(X.columns[1], size=14)
# plt.title('SVM Decision Region Boundary', size=16)

# plt.xlabel('Domestic Performance')
# plt.ylabel('INTL')
# plt.legend()
# plt.show()
