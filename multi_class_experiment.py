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
from imblearn.over_sampling import SMOTE


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
        class_interval = ( np.max(wc_performance[:,0]) - np.min(wc_performance[:,0]) ) / 3
        for performance in wc_performance:
            if(performance < class_interval):
                post_performance.append("C")
            elif (performance < (class_interval*2)):
                post_performance.append("B")
            else:
                post_performance.append("A")
        print("Printing WC performances")
        print(post_performance)
        print("Done")
        # exit()
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


def initialise():
    # Connect to the database
    mean_performance = 0
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
    
    # print("Mean")
    # mean_performance = sum(performance_list_2011[:, 0])/len(performance_list_2011)
    # print(mean_performance)
    # mean_performance = sum(performance_list_2015[:, 0])/len(performance_list_2015)
    # print(mean_performance)
    # mean_performance = sum(performance_list_2013[:, 0])/len(performance_list_2013)
    # print(mean_performance)
    # mean_performance = sum(performance_list_2017[:, 0])/len(performance_list_2017)
    # print(mean_performance)

    # class_interval = np.max()

    np_players = np.concatenate(
        (player_list_2011, player_list_2013, player_list_2015, player_list_2017), axis=0)
    np_performances = np.concatenate(
        (performance_list_2011, performance_list_2013, performance_list_2015, performance_list_2017), axis=0)

    # print(np_performances)
    # class_interval = ( np.max(np_performances[:,0]) - np.min(np_performances[:,0]) ) / 3
    # print("class interval")
    # print(class_interval)
    # exit()

    max_career = np.max(np_players[:, 0])
    max_recent = np.max(np_players[:, 1])
    max_away = np.max(np_players[:, 2])
    max_home = np.max(np_players[:, 3])

    np_players = scale_features(
        np_players, max_career, max_recent, max_away, max_home)
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()  
    # np_players = sc.fit_transform(np_players)  
    # X_test = sc.transform(X_test)  

    sm = SMOTE(random_state=41)
    np_players_resampled, np_performances_resampled = sm.fit_resample(np_players, np_performances)
    

    # DO train test split using SKLEARN
    feature_train, feature_test, target_train, target_test = train_test_split(
        np_players_resampled, np_performances_resampled, test_size=0.30, random_state=42)

    # pca = PCA(n_components=2)  
    # feature_train = pca.fit_transform(feature_train)  
    # feature_test = pca.transform(feature_test)  

    # Train Naive Bayes model
    # gnb = GaussianNB()
    # gnb.fit(feature_train, target_train)
    # nb_pred_prob = gnb.predict_proba(feature_test)
    # nb_pred = gnb.predict(feature_test)
    # print(classification_report(target_test, nb_pred))
    # acc = accuracy_score(nb_pred, target_test)
    # print(acc)

    # exit()

    # Train Multi-layer Perceptron model
    # mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-06,
    #                         hidden_layer_sizes=(13), random_state=7, max_iter=1100)
    # mlp_clf.fit(feature_train, target_train)
    # mlp_pred_prob = mlp_clf.predict_proba(feature_test)
    # mlp_pred = mlp_clf.predict(feature_test)
    # print(classification_report(target_test, mlp_pred))
    # acc = accuracy_score(mlp_pred, target_test)
    # print(acc)

    # exit()
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
    #                  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
    #                 {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
    #                  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
    #                 {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
    #                ]

    # scores = ['precision', 'recall']

    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()

    #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
    #                     scoring='%s_macro' % score)
    #     clf.fit(feature_train, target_train)

    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #             % (mean, std * 2, params))
    #     print()
    

    # parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    # clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
    # clf.fit(feature_train, target_train)
    # print(clf.score(feature_test, target_test))
    # print(clf.best_params_)
              

    # Train SVM model
    # svm_clf = SVC(C=1000, kernel='sigmoid', gamma=0.001, probability=True)
    # svm_clf.fit(feature_train, target_train)
    # svm_pred = svm_clf.predict(feature_test)
    # svm_pred_prob = svm_clf.predict_proba(feature_test)
    # print(classification_report(target_test, svm_pred))
    # acc = accuracy_score(svm_pred, target_test)
    # print(acc)

    # exit()

    # Train Decision Tree model
    desT = DecisionTreeClassifier(max_depth=11)
    desT.fit(feature_train, target_train)
    # desc_pred = desT.predict(feature_test)
    # desc_pred_prob = desT.predict_proba(feature_test)
    # print(classification_report(target_test, desc_pred))
    # acc = accuracy_score(desc_pred, target_test)
    # print(acc)

    # exit()

    # amt_say_nb = acceptance_rate(nb_pred, target_test)

    # amt_say_mlp = acceptance_rate(mlp_pred, target_test)

    # amt_say_svm = acceptance_rate(svm_pred, target_test)

    # amt_say_desc = acceptance_rate(desc_pred, target_test)

    # print('Amount of say NB :', amt_say_nb)
    # print('Amount of say MLP :', amt_say_mlp)
    # print('Amount of say SVM :', amt_say_svm)
    # print('Amount of say Descision Tree :', amt_say_desc)

    return connection, desT, max_home, max_away, max_recent, max_career, feature_train, feature_test, target_train, target_test

def prediction_class_based(players):
    prediction = desT.predict(players)
    return prediction

connection, desT, max_home, max_away, max_recent, max_career, feature_train, feature_test, target_train, target_test = initialise()
