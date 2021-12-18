"""
Note: This project is still in progress. As my computer was damaged, much of my work remains unretreivable.
"""

import pandas as pd
import os 
import warnings
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor 
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('frauddetection.csv')

df.head()

X = df.iloc[:,[2,4,5,7,8]]

y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

"""kNN"""

knnreg = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
knn_ypred = knnreg.predict(X_test)

print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_ypred)))
print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_ypred)))
print("\n")
print('Precision (positive) score of the K-Nearest Neighbors model is {}'.format(metrics.precision_score(y_test, knn_ypred, pos_label=1)))
print('Precision (negative) score of the K-Nearest Neighbors model is {}'.format(metrics.precision_score(y_test, knn_ypred, pos_label=0)))
print('Specificity score of the K-Nearest Neighbors model is {}'.format(metrics.recall_score(y_test, knn_ypred, pos_label=0)))
print('Sensitivity score of the K-Nearest Neighbors model is {}'.format(metrics.recall_score(y_test, knn_ypred, pos_label=1)))

grid_params = {'n_neighbors':[3,5,11,19], 'weights':['uniform', 'distance'], 'metric':['euclidean', 'manhattan']}
clf = GridSearchCV(KNeighborsClassifier(), param_grid = grid_params, cv = 5, refit = True, verbose=2)
clf.fit(X_train, y_train)
print(clf.best_estimator_)

"""Logistic Regression"""

logreg = LogisticRegression().fit(X_train, y_train)
logreg_ypred = logreg.predict(X_test)

print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, logreg_ypred)))
print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, logreg_ypred)))
print("\n")
print('Precision (positive) score of the Logistic Regression model is {}'.format(metrics.precision_score(y_test, logreg_ypred, pos_label=1)))
print('Precision (negative) score of the Logistic Regression model is {}'.format(metrics.precision_score(y_test, logreg_ypred, pos_label=0)))
print('Specificity score of the Logistic Regression model is {}'.format(metrics.recall_score(y_test, logreg_ypred, pos_label=0)))
print('Sensitivity score of the Logistic Regression model is {}'.format(metrics.recall_score(y_test, logreg_ypred, pos_label=1)))

"""SVM (RBF)"""

svm = SVC().fit(X_train, y_train)
svm_ypred = svm.predict(X_test)

print('Accuracy score of the SVM (RBF) model is {}'.format(accuracy_score(y_test, svm_ypred)))
print('F1 score of the SVM (RBF) model is {}'.format(f1_score(y_test, svm_ypred)))
print("\n")
print('Precision (positive) score of the SVM (RBF) model is {}'.format(metrics.precision_score(y_test, svm_ypred, pos_label=1)))
print('Precision (negative) score of the SVM (RBF) model is {}'.format(metrics.precision_score(y_test, svm_ypred, pos_label=0)))
print('Specificity score of the SVM (RBF) model is {}'.format(metrics.recall_score(y_test, svm_ypred, pos_label=0)))
print('Sensitivity score of the SVM (RBF) model is {}'.format(metrics.recall_score(y_test, svm_ypred, pos_label=1)))

"""SVM (Linear)"""

svml = SVC(kernel="linear").fit(X_train, y_train)
svml_ypred = svml.predict(X_test)

print('Accuracy score of the SVM (Linear) model is {}'.format(accuracy_score(y_test, svml_ypred)))
print('F1 score of the SVM (Linear) model is {}'.format(f1_score(y_test, svml_ypred)))
print("\n")
print('Precision (positive) score of the SVM (Linear) model is {}'.format(metrics.precision_score(y_test, svml_ypred, pos_label=1)))
print('Precision (negative) score of the SVM (Linear) model is {}'.format(metrics.precision_score(y_test, svml_ypred, pos_label=0)))
print('Specificity score of the SVM (Linear) model is {}'.format(metrics.recall_score(y_test, svml_ypred, pos_label=0)))
print('Sensitivity score of the SVM (Linear) model is {}'.format(metrics.recall_score(y_test, svml_ypred, pos_label=1)))

"""Random Forest"""

rf = RandomForestClassifier(max_depth = 4).fit(X_train, y_train)
rf_ypred = rf.predict(X_test)

print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_ypred)))
print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_ypred)))
print("\n")
print('Precision (positive) score of the Random Forest model is {}'.format(metrics.precision_score(y_test, rf_ypred, pos_label=1)))
print('Precision (negative) score of the Random Forest model is {}'.format(metrics.precision_score(y_test, rf_ypred, pos_label=0)))
print('Specificity score of the Random Forest model is {}'.format(metrics.recall_score(y_test, rf_ypred, pos_label=0)))
print('Sensitivity score of the Random Forest model is {}'.format(metrics.recall_score(y_test, rf_ypred, pos_label=1)))

"""XGBoost"""

xgb = XGBClassifier(max_depth = 4).fit(X_train, y_train)
xgb_ypred = xgb.predict(X_test)

print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_ypred)))
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_ypred)))
print("\n")
print('Precision (positive) score of the XGBoost model is {}'.format(metrics.precision_score(y_test, xgb_ypred, pos_label=1)))
print('Precision (negative) score of the XGBoost model is {}'.format(metrics.precision_score(y_test, xgb_ypred, pos_label=0)))
print('Specificity score of the XGBoost model is {}'.format(metrics.recall_score(y_test, xgb_ypred, pos_label=0)))
print('Sensitivity score of the XGBoost model is {}'.format(metrics.recall_score(y_test, xgb_ypred, pos_label=1)))

"""Currently, it appears that kNN classifier gives the best results.
I am still working on improving the model by using neural networks that could help to determine if a customer's past transactions could help determine if a current transaction is a fraud.
"""
