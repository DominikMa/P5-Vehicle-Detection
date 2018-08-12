"""Create and train the classifiers."""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import time


def __create_svm(x_train, y_train, parameter):
    parameter_count = []
    for p in parameter:
        for _, item in p.items():
            parameter_count.append(len(item))
    if max(parameter_count) > 1 or len(parameter) > 1:
        search = True
        clf = GridSearchCV(LinearSVC(), parameter,
                           n_jobs=2, verbose=0)
        print("Search for best SVM...")
    else:
        search = False
        parameter = parameter[0]
        p = {}
        for key, item in parameter.items():
            p[key] = item[0]
            
        clf = LinearSVC(**p)
        print("Fit SVM...")

    t1 = time.time()
    clf.fit(x_train, y_train)
    t2 = time.time()
    if search:
        print("Found a SVM.")
        print("Best parameters set found:")
        print(clf.best_params_)
        print("Duration:", round(t2-t1, 2), 'Seconds for', len(x_train),
              'samples and', len(clf.cv_results_['params']),
              'parameter combinations')
    else:
        print("Duration:", round(t2-t1, 2), 'Seconds for', len(x_train),
              'samples')
    print()
    return clf


def __create_dt(x_train, y_train, parameter):
    parameter_count = []
    for p in parameter:
        for _, item in p.items():
            parameter_count.append(len(item))
    if max(parameter_count) > 1 or len(parameter) > 1:
        search = True
        clf = GridSearchCV(DecisionTreeClassifier(), parameter,
                           n_jobs=-1, verbose=0)
        print("Search for best DT...")
    else:
        search = False
        parameter = parameter[0]
        p = {}
        for key, item in parameter.items():
            p[key] = item[0]
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**p),
                                 n_estimators=10)
        print("Fit DT...")

    t1 = time.time()
    clf.fit(x_train, y_train)
    t2 = time.time()
    if search:
        print("Found a DT.")
        print("Best parameters set found:")
        print(clf.best_params_)
        print("Duration:", round(t2-t1, 2), 'Seconds for', len(x_train),
              'samples and', len(clf.cv_results_['params']),
              'parameter combinations')
    else:
        print("Duration:", round(t2-t1, 2), 'Seconds for', len(x_train),
              'samples')
    print()
    return clf


def create_classifier(x_train_svm, x_train_dt, y_train, settings):
    """Find and return the best SVM and DT for the training data."""
    if settings['find_svm']:
        parameter_svm = settings['parameter_svm']
        svm = __create_svm(x_train_svm, y_train, parameter_svm)
    else:
        svm = None
    if settings['find_dt']:
        parameter_dt = settings['parameter_dt']
        dt = __create_dt(x_train_dt, y_train, parameter_dt)
    else:
        dt = None

    return svm, dt
