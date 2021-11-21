"""
ML Assignment 2
Author: Rajarshi Kesh
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


project_level_directory = ""


class DataReader:
    def __init__(self, path, x_cols, y_col, header=None):
        self.df = pd.read_csv(path, header=header)
        self.X = self.df.iloc[:, x_cols[0]:x_cols[1]]
        self.y = self.df.iloc[:, y_col]

    @staticmethod
    def displayData(path_info, path_data, headers=None):
        with open(path_info) as f:
            print(f.read())
        df = pd.read_csv(path_data, header=headers)
        print(df.head())

    def preProcess(self, test_size=0.3, label=False, scale=False, pca=False):
        if label:
            self.y = LabelEncoder().fit_transform(self.y)

        print(f"Train Test Split: {int((1 - test_size) * 100)}/{int(test_size * 100)}")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=8)

        if scale:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        if pca:
            pca = PCA(n_components='mle')
            print(f"Previous no. of features: {X_train.shape[1]}")
            X_train = pca.fit_transform(X_train)
            print(f"New no. of features: {X_train.shape[1]}")
            X_test = pca.transform(X_test)

        return X_train, X_test, y_train, y_test


class Model:
    def __init__(self, X_train, X_test, y_train, y_test, model_key, **kwargs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model_key
        models = {
            'svm': SVC,
            'mlp': MLPClassifier,
            'dtc': DecisionTreeClassifier,
            'rfc': RandomForestClassifier
        }
        self.classifier = models[model_key](**kwargs)
        self.tuned_classifier = None

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)

    def test_model(self, tunedModel=False):
        model = self.classifier if not tunedModel else self.tuned_classifier
        print("------------------------------------------")
        print(f"Model: {model}")
        print('-----------------------------------')
        y_pred = model.predict(self.X_test)
        print("Confusion Matrix")
        print(confusion_matrix(self.y_test, y_pred))

        print('-----------------------------------')
        print('-----------------------------------')

        print('Performance Evaluation:')
        print(classification_report(self.y_test, y_pred))

        print('-----------------------------------')
        print('-----------------------------------')

        print('Accuracy Score:', end=" ")
        print(accuracy_score(self.y_test, y_pred) * 100, end="%\n")

        print('-----------------------------------')

        plot_confusion_matrix(model, self.X_test, self.y_test)
        plt.title('Heat map for confusion matrix')
        plt.savefig(f'{self.model}_conf' + ('_tuned' if tunedModel else ''))

        y_p_prob = model.predict_proba(self.X_test)
        skplt.metrics.plot_roc(self.y_test, y_p_prob)
        plt.savefig(f'{self.model}_roc' + ('_tuned' if tunedModel else ''))

    @ignore_warnings(category=ConvergenceWarning)
    def parameter_tuning(self, param_grid):
        self.tuned_classifier = self.classifier
        grid_search = GridSearchCV(estimator=self.tuned_classifier, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        print("Parameter Tuning Completed.")
        print(f"Best Parameters: {grid_search.best_params_}")
        self.tuned_classifier = grid_search.best_estimator_


# Parameter Grids for Parameter Tuning
param_grid = {
    'svm': {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
    },
    'mlp': {
        'max_iter': [500, 750, 1000],
        'momentum': [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    },
    'dtc': {

    },
    'rfc': {
        'bootstrap': [True, False],
        'max_depth': [10, 30, 50, 70, 85, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [50, 100, 200]
    }
}


# DataReader.displayData('data/breast-cancer-wisconsin.names', 'data/bcancer.data')

# df = pd.read_csv('data/breast-cancer-wisconsin.data', header=None)
# df = df[df[6] != '?']
#
# df.to_csv('data/bcancer.data', header=False, index=False)

test_size_list = [0.3]
data_key_list = ['wine', 'iris', 'ion', 'bcancer']

# , 'iris', 'ion', 'bcancer'

# 'wine', 'iris', 'ion',
test_data = {'wine': {'path': 'data/wine.data', 'x_cols': (1, 14), 'y_col': 0},
             'iris': {'path': 'data/iris.data', 'x_cols': (0, 4), 'y_col': 4},
             'ion': {'path': 'data/ionosphere.data', 'x_cols': (1, 34), 'y_col': 34},
             'bcancer': {'path': 'data/bcancer.data', 'x_cols': (1, 10), 'y_col': 10}}

for data_key in data_key_list:
    for t_size in test_size_list:
        os.chdir(f'{project_level_directory}')
        data = DataReader(**test_data[data_key])
        print(f"{data_key.upper()} DATASET")
        print("-------------------------------")

        params = {'pca': True}
        if data_key == 'wine':
            params['scale'] = True

        X_train, X_test, y_train, y_test = data.preProcess(test_size=t_size, **params)

        Path(f"{project_level_directory}/graphs/{data_key}/{t_size}").mkdir(parents=True, exist_ok=True)
        os.chdir(f"{project_level_directory}/graphs/{data_key}/{t_size}")

        # iris_data = DataReader('data/iris.data', (0, 4), 4)
        # print("IRIS DATASET")
        # print("-------------------------------")
        # X_train, X_test, y_train, y_test = iris_data.preProcess(scale=True)

        # ion_data = DataReader('data/ionosphere.data', (0, 34), 34)
        # print("ION DATASET")
        # print("-------------------------------")
        # X_train, X_test, y_train, y_test = ion_data.preProcess()

        # bcancer_data = DataReader('data/breast-cancer-wisconsin.data', (0,2), 3)
        # print("BREAST CANCER DATASET")
        # print("-------------------------------")
        # X_train, X_test, y_train, y_test = bcancer_data.preProcess()

        svmModel = Model(X_train, X_test, y_train, y_test, 'svm', probability=True)
        svmModel.train_model()
        svmModel.test_model()
        svmModel.parameter_tuning(param_grid['svm'])
        svmModel.test_model(tunedModel=True)

        mlpModel = Model(X_train, X_test, y_train, y_test, 'mlp', max_iter=750)
        mlpModel.train_model()
        mlpModel.test_model()
        mlpModel.parameter_tuning(param_grid['mlp'])
        mlpModel.test_model(tunedModel=True)

        rfcModel = Model(X_train, X_test, y_train, y_test, 'rfc')
        rfcModel.train_model()
        rfcModel.test_model()
        rfcModel.parameter_tuning(param_grid['rfc'])
        rfcModel.test_model(tunedModel=True)
