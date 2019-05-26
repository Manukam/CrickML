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
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
from Classes.interntional_player import International_Player
from domestic_model import domestic_model_initialise
from domestic_model import build_domestic_player
from domestic_model import build_domestic_features
from domestic_model import get_domestic_predictions
from domestic_model import scale_domestic_features
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

app = Flask(__name__)
# cors = CORS(app, resources={r"/selectedPlayers": {"origins": "*"}})
CORS(app)


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

                pre_performance.append(
                    [career_score, recent_score, away_score, home_score])
    finally:
        print('done')
        return pre_performance


def fetch_data_post(query, connection):
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


def scale_features(players, max_career, max_recent, max_away, max_home):
    # np_player_list = np.array(players)

    for player in players:
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

    return players


def acceptance_rate(prediction, test_data):
    error_rate = 0
    for index, pred in enumerate(prediction):
        if(pred != test_data[index]):
            error_rate += 1
    return 1/2 * (math.log((1-(error_rate/len(prediction)))/(error_rate/len(prediction))))


def initialise():
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='crickml',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    # Fetch data for features

    print("Fetching player data for Features.....")

    player_list_2011 = fetch_data_pre('SELECT * FROM pre_wc_2011', connection)
    player_list_2015 = fetch_data_pre('SELECT * FROM pre_wc_2015', connection)
    player_list_2013 = fetch_data_pre('SELECT * FROM pre_ct_2013', connection)
    player_list_2017 = fetch_data_pre('SELECT * FROM pre_ct_2017', connection)
    # player_list_2007 = fetch_data_pre('SELECT * FROM pre_ct_2007', connection)

    # Fetch data for labels

    print("Fetching player data for Labels.....")

    performance_list_2011 = fetch_data_post(
        'SELECT * FROM wc_2011', connection)
    performance_list_2015 = fetch_data_post(
        'SELECT * FROM wc_2015', connection)
    performance_list_2013 = fetch_data_post(
        'SELECT * FROM ct_2013', connection)
    performance_list_2017 = fetch_data_post(
        'SELECT * FROM ct_2017', connection)
    # performance_list_2007 = fetch_data_post(
    #     'SELECT * FROM wc_2007', connection)

    np_players = np.concatenate(
        (player_list_2011, player_list_2013, player_list_2015, player_list_2017), axis=0)
    np_performances = np.concatenate(
        (performance_list_2011, performance_list_2013, performance_list_2015, performance_list_2017), axis=0)

    max_career = np.max(np_players[:, 0])
    max_recent = np.max(np_players[:, 1])
    max_away = np.max(np_players[:, 2])
    max_home = np.max(np_players[:, 3])

    np_players = scale_features(
        np_players, max_career, max_recent, max_away, max_home)

    # DO train test split using SKLEARN
    feature_train, feature_test, target_train, target_test = train_test_split(
        np_players, np_performances, test_size=0.30, random_state=42)

    # Train Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(feature_train, target_train)
    # nb_pred_prob = gnb.predict_proba(feature_test)
    nb_pred = gnb.predict(feature_test)

    # Train Multi-layer Perceptron model
    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
    mlp_clf.fit(feature_train, target_train)
    # mlp_pred_prob = mlp_clf.predict_proba(feature_test)
    mlp_pred = mlp_clf.predict(feature_test)

    # Train SVM model
    svm_clf = SVC(C=1000, kernel='sigmoid', gamma=0.001, probability=True)
    svm_clf.fit(feature_train, target_train)
    svm_pred = svm_clf.predict(feature_test)
    # svm_pred_prob = svm_clf.predict_proba(feature_test)

    # Train Decision Tree model
    desT = DecisionTreeClassifier(max_depth=2)
    desT.fit(feature_train, target_train)
    desc_pred = desT.predict(feature_test)
    # desc_pred_prob = desT.predict_proba(feature_test)

    amt_say_nb = acceptance_rate(nb_pred, target_test)

    amt_say_mlp = acceptance_rate(mlp_pred, target_test)

    amt_say_svm = acceptance_rate(svm_pred, target_test)

    amt_say_desc = acceptance_rate(desc_pred, target_test)

    print('Amount of say NB :', amt_say_nb)
    print('Amount of say MLP :', amt_say_mlp)
    print('Amount of say SVM :', amt_say_svm)
    print('Amount of say Descision Tree :', amt_say_desc)

    return connection, gnb, mlp_clf, svm_clf, desT, amt_say_desc, amt_say_mlp, amt_say_nb, amt_say_svm, max_home, max_away, max_recent, max_career, feature_train, feature_test, target_train, target_test


connection, nb, mlp, svm, dest, desc_say, mlp_say, nb_say, svm_say, max_home, max_away, max_recent, max_career, feature_train, feature_test, target_train, target_test = initialise()


def get_player_pool(connection):
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM sri_lanka')
        result = cursor.fetchall()
        return result


def built_features(player):

    career_score = player.calculate_overall_score()

    away_score = player.calculate_away_score()

    home_score = player.calculate_home_score()

    recent_score = player.calculate_recent_score()

    return [career_score, recent_score, away_score, home_score]


prediction_proba = []


def prediction_engine(player_list):
    nb_pred_prob = nb.predict_proba(player_list)
    # nb_pred = nb.predict(player_list)
    mlp_pred_prob = mlp.predict_proba(player_list)
    # mlp_pred = mlp.predict(player_list)
    # svm_pred = svm.predict(player_list)
    svm_pred_prob = svm.predict_proba(player_list)
    # desc_pred = dest.predict(player_list)
    desc_pred_prob = dest.predict_proba(player_list)

    return nb_pred_prob, mlp_pred_prob, svm_pred_prob, desc_pred_prob


def crickml_algo(nb_pred_prob, mlp_pred_prob, svm_pred_prob, desc_pred_prob):

    final_predictions = []

    for index, initial_nb_pred in enumerate(nb_pred_prob):
        print(initial_nb_pred)
        weighted_nb_prediction0 = nb_say * (nb_pred_prob[index][0])
        weighted_nb_prediction1 = nb_say * (nb_pred_prob[index][1])

        weighted_mlp_prediction0 = mlp_say * (mlp_pred_prob[index][0])
        weighted_mlp_prediction1 = mlp_say * (mlp_pred_prob[index][1])

        weighted_svm_prediction0 = svm_say * (svm_pred_prob[index][0])
        weighted_svm_prediction1 = svm_say * (svm_pred_prob[index][1])

        weighted_desc_prediction0 = desc_say * (desc_pred_prob[index][0])
        weighted_desc_prediction1 = desc_say * (desc_pred_prob[index][1])

        mean_weighted_prediction0 = (weighted_mlp_prediction0 + weighted_nb_prediction0 +
                                     weighted_svm_prediction0 + weighted_desc_prediction0) / 4
        mean_weighted_prediction1 = (weighted_mlp_prediction1 + weighted_nb_prediction1 +
                                     weighted_svm_prediction1 + weighted_desc_prediction1) / 4

        prediction_proba.append(mean_weighted_prediction1)
        print(mean_weighted_prediction0)
        print(mean_weighted_prediction1)

        if(mean_weighted_prediction0 > mean_weighted_prediction1):
            confidence = (nb_pred_prob[index][0] + mlp_pred_prob[index]
                          [0] + svm_pred_prob[index][0] + desc_pred_prob[index][0]) / 4
            final_predictions.append([0, confidence])
        else:
            confidence = (nb_pred_prob[index][1] + mlp_pred_prob[index]
                          [1] + svm_pred_prob[index][1] + desc_pred_prob[index][1]) / 4
            final_predictions.append([1, confidence])

    return final_predictions


# Build players with provided data.
def build_player(player):
    print("Building player...")
    in_player = International_Player(player["id"], player['player_name'], player['overall_matches'], player['overall_innings'], player['overall_runs'], player['overall_average'],
                                     player['overall_strike_rate'], player['overall_100s'], player['overall_50s'], player[
                                         'home_matches'], player['home_innings'], player['home_runs'], player['home_average'],
                                     player['home_strike_rate'], player['home_100s'], player['home_50s'], player['away_matches'], player[
        'away_innings'], player['away_runs'], player['away_average'], player['away_strike_rate'],
        player['away_100s'], player['away_50s'], player['form_matches'], player['form_innings'], player['form_runs'], player['form_average'], player['form_strike_rate'], player['form_100s'], player['form_50s'])
    return in_player


def analyse_players(players):
    selected_players = []
    # selected_players_domestic = []
    with connection.cursor() as cursor:
        for player in players:
            sql = ('select * from sri_lanka where id=%s')
            cursor.execute(sql, player)
            result = cursor.fetchall()
            if(result[0]['overall_matches'] < 5):
                domestic_player = build_domestic_player(result[0])
                domestic_player_features = build_domestic_features(
                    domestic_player)
                scaled_domestic_features = scale_domestic_features(
                    domestic_player_features)
                nb_pred_prob_dom, mlp_pred_prob_dom, svm_pred_prob_dom, desc_pred_prob_dom = get_domestic_predictions(
                    scaled_domestic_features)
            else:
                player = build_player(result[0])
                selected_players.append(built_features(player))

        selected_players = scale_features(
            selected_players, max_career, max_recent, max_away, max_home)
        nb_pred_prob, mlp_pred_prob, svm_pred_prob, desc_pred_prob = prediction_engine(
            selected_players)
        total_predictions = crickml_algo(
            nb_pred_prob, mlp_pred_prob, svm_pred_prob, desc_pred_prob)
        total_predictions.append(crickml_algo(
            nb_pred_prob_dom, mlp_pred_prob_dom, svm_pred_prob_dom, desc_pred_prob_dom))
        # print(players[0])
        # exit()
        result_array = [[0 for x in range(7)]
                        for y in range(len(total_predictions))]
        print(len(result_array))
        # exit()
        for index, prediction in enumerate(total_predictions):
            result_array[index][0] = players[index]
            result_array[index][1] = prediction[0]
            result_array[index][2] = prediction[1]
            result_array[index][3] = selected_players[index][0]
            result_array[index][4] = selected_players[index][1]
            result_array[index][5] = selected_players[index][2]
            result_array[index][6] = selected_players[index][3]
        return result_array


def generate_roc():
    prediction_engine(feature_test)
    auc = roc_auc_score(target_test, prediction_proba)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(target_test, prediction_proba)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()


# generate_roc()


# API Method to iniialise the UI with all the players
@app.route('/players', methods=['GET'])
def get_players():
    response = jsonify(get_player_pool(connection))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# API Method to accept the selected players and return output predictions and results.
@app.route('/selected_players/', methods=['POST'])
# @cross_origin(origin='**')
def selected_players():
    response = jsonify(analyse_players(request.get_json()))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    app.run(debug=True)
