import pymysql.cursors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import math as math
from sklearn import preprocessing
from flask import Flask
from flask import jsonify

app = Flask(__name__)


def batsmen_model(matches, innings, average, hundreds, fifties):
    if(innings <= 0):
        return 0.0
    else:
        u = innings/matches
        v = (20 * hundreds) + (5 * fifties)
        w = (0.3 * v) + (0.7 * average)
        return u * w


def batsmen_model_form(matches, innings, average, runs, hundreds, fifties, strike_rate):
    if(innings <= 0):
        return 0.0
    else:
        u = innings/matches
        v = (20 * hundreds) + (5 * fifties)
        w = (0.5 * v) + (0.2 * strike_rate) + (0.5 * runs)
        return u * w


def fetch_data_pre(query, connection):
    try:
        with connection.cursor() as cursor:
            # Read a single record
            pre_performance = []
            cursor.execute(query)
            result = cursor.fetchall()
            for player in result:
                career_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                            player['overall_average'], player['overall_100s'], player['overall_50s'])

                away_score = batsmen_model(player['away_matches'], player['away_innings'],
                                        player['away_average'], player['away_100s'], player['away_50s'])

                home_score = batsmen_model(player['home_matches'], player['home_innings'],
                                        player['home_average'], player['home_100s'], player['home_50s'])

                # recent_score = batsmen_model_form(player['form_matches'], player['form_innings'],
                #                            player['form_average'], player['form_runs'] , player['form_100s'], player['form_50s'], player['form_strike_rate'])

                recent_score = batsmen_model(player['form_matches'], player['form_innings'],
                                            player['form_average'], player['form_100s'], player['form_50s'])

                # career_score = career_score + condition_performances

                # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

                pre_performance.append([career_score, recent_score, away_score, home_score])
    finally:
        print('done')
        return pre_performance

def fetch_data_post(query,connection):
    with connection.cursor() as cursor:
        # Read a single record
        # sql = query
        cursor.execute(query)
        result = cursor.fetchall()
        performance = []
        post_performance = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            performance.append([tournement_score])
            # performance_list.append([tournement_score])

        wc_performance = np.array(performance)
        mean_performance = sum(wc_performance[:, 0])/len(performance)
        for performance in wc_performance:
            if(performance < mean_performance):
                post_performance.append(0)
            else:
                post_performance.append(1)
    
    return post_performance


def initialise():
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='crickml',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
    player_list_2011 = []
    player_list_2013 = []
    player_list_2015 = []
    player_list_2017 = []

    performance_list_2011 = []
    performance_list_2013 = []
    performance_list_2015 = []
    performance_list_2017 = []

    player_list_2011 = fetch_data_pre('SELECT * FROM pre_wc_2011',connection)
    player_list_2015 = fetch_data_pre('SELECT * FROM pre_wc_2015',connection)
    player_list_2013 = fetch_data_pre('SELECT * FROM pre_ct_2013',connection)
    player_list_2017 = fetch_data_pre('SELECT * FROM pre_ct_2017',connection)

    performance_list_2011 = fetch_data_post('SELECT * FROM wc_2011',connection)
    performance_list_2015 = fetch_data_post('SELECT * FROM wc_2015',connection)
    performance_list_2013 = fetch_data_post('SELECT * FROM ct_2013',connection)
    performance_list_2017 = fetch_data_post('SELECT * FROM ct_2017',connection)

    np_players = np.concatenate((player_list_2011,player_list_2013,player_list_2015,player_list_2017), axis=0)
    np_performances = np.concatenate((performance_list_2011,performance_list_2013,performance_list_2015,performance_list_2017), axis=0)

    max_career = np.max(np_players[:, 0])
    max_recent = np.max(np_players[:, 1])
    max_away = np.max(np_players[:, 2])
    max_home = np.max(np_players[:, 3])

    for player in np_players:
        career_score = player[0]
        recent_score = player[1]
        away_score = player[2]
        home_score = player[3]

        normalized_career_score = career_score/max_career
        player[0] = normalized_career_score

        if(recent_score != 0):
            normalized_recent_score = recent_score/max_recent
            player[1] = normalized_recent_score

        if(away_score != 0):
            normalized_away_score = away_score/max_away
            player[2] = normalized_away_score

        if(home_score != 0):
            normalized_home_score = home_score/max_home
            player[3] = normalized_home_score


    feature_train, feature_test, target_train, target_test = train_test_split(
        np_players, np_performances, test_size=0.20, random_state=42)


    gnb = GaussianNB()
    gnb.fit(feature_train, target_train)
    # nb_pred_prob = gnb.predict_proba(feature_test)
    nb_pred = gnb.predict(feature_test)

    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp_clf.fit(feature_train, target_train)
    # mlp_pred_prob = mlp_clf.predict_proba(feature_test)
    mlp_pred = mlp_clf.predict(feature_test)

    svm_clf = SVC(C=1000, kernel='sigmoid', gamma=0.001, probability=True)
    svm_clf.fit(feature_train, target_train)
    svm_pred = svm_clf.predict(feature_test)
    # svm_pred_prob = svm_clf.predict_proba(feature_test)

    desT = DecisionTreeClassifier(max_depth=2)
    desT.fit(feature_train, target_train)
    desc_pred = desT.predict(feature_test)
    # desc_pred_prob = desT.predict_proba(feature_test)

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

    return connection, gnb, mlp_clf, svm_clf, desT, amt_say_desc, amt_say_mlp, amt_say_nb, amt_say_svm

def get_player_pool(connection):
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM sri_lanka')
        result = cursor.fetchall()
        return result

connection, nb, mlp, svm, dest, desc_say, mlp_say, nb_say, svm_say = initialise()


# print('NB Pred :', nb_pred)
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


@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/players')
def get_players():
  return jsonify(get_player_pool(connection)) 

@app.route('/selectedPlayers/<players>')
def analyse_players(players):
    print('fuck')

if __name__ == '__main__':
    app.run()