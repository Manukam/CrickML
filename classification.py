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

# np.set_printoptions(suppress=True)


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

            # recent_score = batsmen_model_form(player['form_matches'], player['form_innings'],
            #                            player['form_average'], player['form_runs'] , player['form_100s'], player['form_50s'], player['form_strike_rate'])

            recent_score = batsmen_model(player['form_matches'], player['form_innings'],
                                         player['form_average'], player['form_100s'], player['form_50s'])

            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append(
                [career_score, recent_score, away_score, home_score])

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

            # recent_score = batsmen_model_form(player['form_matches'], player['form_innings'],
            #                            player['form_average'], player['form_runs'] , player['form_100s'], player['form_50s'], player['form_strike_rate'])

            recent_score = batsmen_model(player['form_matches'], player['form_innings'],
                                         player['form_average'], player['form_100s'], player['form_50s'])
            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append(
                [career_score, recent_score, away_score, home_score])

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

    #         recent_score = batsmen_model_form(player['form_matches'], player['form_innings'],
    #                                    player['form_average'], player['form_runs'] , player['form_100s'], player['form_50s'], player['form_strike_rate'])

    #         player_list.append([career_score, recent_score, away_score, home_score])

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

            # recent_score = batsmen_model_form(player['form_matches'], player['form_innings'],
            #                            player['form_average'], player['form_runs'] , player['form_100s'], player['form_50s'], player['form_strike_rate'])

            recent_score = batsmen_model(player['form_matches'], player['form_innings'],
                                         player['form_average'], player['form_100s'], player['form_50s'])
            # career_score = career_score + condition_performances

            # batsmen_score = (0.35 * career_score) + (0.65 * player['form_average'] )

            player_list.append(
                [career_score, recent_score, away_score, home_score])

        np_players = np.array(player_list)
        # print(np_players)
        # print(np_players.shape)
        # exit()
        max_career = np.max(np_players[:, 0])
        max_recent = np.max(np_players[:, 1])
        max_away = np.max(np_players[:, 2])
        max_home = np.max(np_players[:, 3])
        # print(max_career)
        # print(max_away)
        # print(max_home)
        # exit()

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

    # a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `wc_2011`"
        cursor.execute(sql)
        result = cursor.fetchall()
        performance_list = []
        wc_2011_performance = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            wc_2011_performance.append([tournement_score])
            # performance_list.append([tournement_score])

        wc_2011 = np.array(wc_2011_performance)
        mean_performance = sum(wc_2011[:, 0])/len(wc_2011_performance)

        # print(mean_performance)
        # print(wc_2011)

        for performance in wc_2011:
            if(performance <= mean_performance):
                performance_list.append(0)
            else:
                performance_list.append(1)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `wc_2015`"
        cursor.execute(sql)
        result = cursor.fetchall()
        wc_2015_performance = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            wc_2015_performance.append([tournement_score])

        wc_2015 = np.array(wc_2015_performance)
        mean_performance = sum(wc_2015[:, 0])/len(wc_2015_performance)

        for performance in wc_2015:
            if(performance <= mean_performance):
                performance_list.append(0)
            else:
                performance_list.append(1)

    # with connection.cursor() as cursor:
    #     # Read a single record
    #     sql = "SELECT * FROM `ct_2017`"
    #     cursor.execute(sql)
    #     result = cursor.fetchall()
    #     ct_2017_performance = []
    #     for player in result:
    #         tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
    #                                          player['overall_average'], player['overall_100s'], player['overall_50s'])
    #         ct_2017_performance.append([tournement_score])

    #     ct_2017 = np.array(ct_2017_performance)
    #     mean_performance = sum(ct_2017[:,0])/len(ct_2017_performance)

    #     for performance in ct_2017:
    #         if(performance <= mean_performance):
    #             performance_list.append(0)
    #         else:
    #             performance_list.append(1)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `ct_2013`"
        cursor.execute(sql)
        result = cursor.fetchall()
        ct_2013_performance = []
        for player in result:
            tournement_score = batsmen_model(player['overall_matches'], player['overall_innings'],
                                             player['overall_average'], player['overall_100s'], player['overall_50s'])
            ct_2013_performance.append([tournement_score])

        ct_2013 = np.array(ct_2013_performance)
        mean_performance = sum(ct_2013[:, 0])/len(ct_2013_performance)

        for performance in ct_2013:
            if(performance <= mean_performance):
                performance_list.append(0)
            else:
                performance_list.append(1)

    np_performances = np.array(performance_list)

finally:
    connection.close()

feature_train, feature_test, target_train, target_test = train_test_split(
    np_players, np_performances, test_size=0.20, random_state=42)

pca = PCA(n_components=2).fit(feature_train)
print(pca.explained_variance_ratio_)
X_train_pca = pca.transform(feature_train)
X_test_pca = pca.transform(feature_test)

clf = SVC(C=1000, kernel='rbf', gamma=0.001)
clf.fit(X_train_pca, target_train)
pred = clf.predict(X_test_pca)
acc = accuracy_score(pred, target_test)
# print(precision_recall_fscore_support(target_test, pred, average='weighted'))
print(classification_report(target_test, pred))
print('Accuracy:', acc)

plot_decision_regions(X=X_train_pca,
                      y=target_train,
                      clf=clf,
                      legend=2)
# print (confusion_matrix(target_test, pred))

# tuned_parameters = {'solver': ['lbfgs'], 'max_iter': [1000], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                      hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train_pca, target_train)
# pred = clf.predict(X_test_pca)
# acc = accuracy_score(pred, target_test)
# print(acc)

# print(precision_recall_fscore_support(target_test, pred, average='weighted'))
# print(acc)


# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['poly'], 'C': [1, 10, 100, 1000]}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

# clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
# clf.fit(X_train_pca, target_train)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#         % (mean, std * 2, params))
# print()

# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# y_true, y_pred = target_test, clf.predict(X_test_pca)
# print(classification_report(y_true, y_pred))
# print()

feature1_for_plot = np_players[:, 0]
feature2_for_plot = np_players[:, 1]


# target_test_for_plot = target_test[:,0]
# target_train_for_plot = target_train[:,0]
# for feature1, feature2, target in np.nditer([feature1_for_plot, feature2_for_plot, np_performances]):
#     if(target == 1):
#         plt.scatter(feature1, feature2, color='r')
#     else:
#         plt.scatter(feature1, feature2, color='b')

# plt.scatter(feature1_for_plot[0],
#             np_performances[0], color='b', label="Below")
# plt.scatter(feature1_for_plot[0],
#             np_performances[0], color='r', label="Above")
# plt.xlabel(X.columns[0], size=14)
# plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)

plt.xlabel('Career Performance')
plt.ylabel('Recent Form')
plt.legend()
plt.show()
