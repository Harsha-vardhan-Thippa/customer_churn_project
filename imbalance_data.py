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
logger = setup_logging("imbalance_data")

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

import pickle

from scipy import stats

from ALL_MODELS import all_models


def balancing_data(x_train,y_train,x_test,y_test):
    try:
        logger.info(f"Before using SMOTE Number of rows that good class Have : {sum(y_train == 1)}")
        logger.info(f"Before using SMOTE Number of rows that bad class Have :  {sum(y_train == 0)}")

        smote = SMOTE(random_state=42)
        x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

        logger.info(f"After using SMOTE Number of rows that good class Have : {sum(y_train_bal == 1)}")
        logger.info(f"After using SMOTE Number of rows that bad class Have :  {sum(y_train_bal == 0)}")

        logger.info(f"shape of the x training data : {x_train_bal.shape}")
        logger.info(f"shape of the y training data : {y_train_bal.shape}")

        scaler = StandardScaler()
        x_train_bal_scaled = scaler.fit_transform(x_train_bal)
        x_test_scaled = scaler.transform(x_test)
        logger.info(f"After scalling the data:\n{x_train_bal_scaled}\n{x_test_scaled}")

        with open("scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

        all_models(x_train_bal_scaled, y_train_bal, x_test_scaled, y_test)


    except Exception as e:
        logger.error(e)