import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging("ALL_MODELS.py")

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

import pickle

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve


def all_models(X_train,y_train,X_test,y_test):
    try:
        def knn(X_train,y_train,X_test,y_test):
          global knn_reg
          knn_reg = KNeighborsClassifier(n_neighbors=5)
          knn_reg.fit(X_train,y_train)
          logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}')

        def nb(X_train,y_train,X_test,y_test):
          global naive_reg
          naive_reg = GaussianNB()
          naive_reg.fit(X_train,y_train)
          logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,naive_reg.predict(X_test))}')

        def lr(X_train,y_train,X_test,y_test):
          global lr_reg
          lr_reg = LogisticRegression()
          lr_reg.fit(X_train,y_train)
          logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test,lr_reg.predict(X_test))}')

        def dt(X_train,y_train,X_test,y_test):
          global dt_reg
          dt_reg = DecisionTreeClassifier(criterion='entropy')
          dt_reg.fit(X_train,y_train)
          logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(X_test))}')

        def rf(X_train,y_train,X_test,y_test):
          global rf_reg
          rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
          rf_reg.fit(X_train,y_train)
          logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(X_test))}')

        def ada(X_train,y_train,X_test,y_test):
          global ada_reg
          t = LogisticRegression()
          ada_reg = AdaBoostClassifier(estimator=t,n_estimators=5)
          ada_reg.fit(X_train,y_train)
          logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test,ada_reg.predict(X_test))}')

        def gb(X_train,y_train,X_test,y_test):
          global gb_reg
          gb_reg = GradientBoostingClassifier(n_estimators=5)
          gb_reg.fit(X_train,y_train)
          logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test,gb_reg.predict(X_test))}')

        def xgb_(X_train,y_train,X_test,y_test):
          global xg_reg
          xg_reg = XGBClassifier()
          xg_reg.fit(X_train,y_train)
          logger.info(f'XGBClassifier Test Accuracy : {accuracy_score(y_test,xg_reg.predict(X_test))}')


        def common(X_train, y_train, X_test, y_test):
            logger.info('=========KNN===========')
            knn(X_train, y_train, X_test, y_test)
            logger.info('=========NB===========')
            nb(X_train, y_train, X_test, y_test)
            logger.info('=========LR===========')
            lr(X_train, y_train, X_test, y_test)
            logger.info('=========DT===========')
            dt(X_train, y_train, X_test, y_test)
            logger.info('=========RF===========')
            rf(X_train, y_train, X_test, y_test)
            logger.info('=========Adabost===========')
            ada(X_train, y_train, X_test, y_test)
            logger.info('=========GB===========')
            gb(X_train, y_train, X_test, y_test)
            logger.info('=========XGB===========')
            xgb_(X_train, y_train, X_test, y_test)

            knn_predictions = knn_reg.predict_proba(X_test)[:, 1]
            naive_predictions = naive_reg.predict_proba(X_test)[:, 1]
            with open("customer_churn.pkl", "wb") as f:
                pickle.dump(naive_reg,f)
            lr_predictions = lr_reg.predict_proba(X_test)[:, 1]
            dt_predictions = dt_reg.predict_proba(X_test)[:, 1]
            rf_predictions = rf_reg.predict_proba(X_test)[:, 1]
            ada_predictions = ada_reg.predict_proba(X_test)[:, 1]
            gb_predictions = gb_reg.predict_proba(X_test)[:, 1]

            xgb_predictions = xg_reg.predict_proba(X_test)[:, 1]


            knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_predictions)
            nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, naive_predictions)
            lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_predictions)
            dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_predictions)
            rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_predictions)
            ada_fpr, ada_tpr, ada_thre = roc_curve(y_test, ada_predictions)
            gb_fpr, gb_tpr, gb_thre = roc_curve(y_test, gb_predictions)
            xgb_fpr, xgb_tpr, xgb_thre = roc_curve(y_test, xgb_predictions)


            plt.figure(figsize=(5, 3))
            plt.plot([0, 1], [0, 1], "k--")

            plt.plot(knn_fpr, knn_tpr, label="KNN")
            plt.plot(nb_fpr, nb_tpr, label="NB")
            plt.plot(lr_fpr, lr_tpr, label="LR")
            plt.plot(dt_fpr, dt_tpr, label="DT")
            plt.plot(rf_fpr, rf_tpr, label="RF")
            plt.plot(ada_fpr, ada_tpr, label="ADA")
            plt.plot(gb_fpr, gb_tpr, label="GB")
            plt.plot(xgb_fpr, xgb_tpr, label="XGB")


            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve - ALL Models")
            plt.legend(loc=0)
            plt.show()

        common(X_train,y_train,X_test,y_test)


    except Exception as ex:
      logger.error(ex)