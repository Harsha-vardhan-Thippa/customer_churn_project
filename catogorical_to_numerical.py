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
logger = setup_logging("categorical_to_numerical")

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def CategoricalToNumerical(x_train_cat,x_test_cat):
    try:
        logger.info(f"Total categorical columns : {x_train_cat.columns}")
        for col in x_train_cat.columns:
            logger.info(f"{col}------>{x_train_cat[col].isnull().sum()}")

        logger.info(f"-----------Unique values of each category-------------- ")
        for col in x_train_cat.columns:
            logger.info(f"{col}------>{x_train_cat[col].unique()}")

        #Nominal encoding
        logger.info("-----------------Nominal Encoding -------------------")
        one_hot_encoder = OneHotEncoder(drop='first')
        one_hot_encoder.fit(x_train_cat[["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","PaperlessBilling","PaymentMethod"]])
        logger.info(f"{one_hot_encoder.get_feature_names_out()}")
        result = one_hot_encoder.transform(x_train_cat[["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","PaperlessBilling","PaymentMethod"]]).toarray()
        # logger.info(f"Result of the nominal Encoding : {result}")
        f = pd.DataFrame(result, columns=one_hot_encoder.get_feature_names_out())
        x_train_cat.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        x_train_cat = pd.concat([x_train_cat, f], axis=1)
        x_train_cat = x_train_cat.drop(["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","PaperlessBilling","PaymentMethod"], axis=1)
        logger.info(f" Train data columns : \n {x_train_cat.columns}")

        result1 = one_hot_encoder.transform(x_test_cat[["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","PaperlessBilling","PaymentMethod"]]).toarray()
        # logger.info(f"Result of the nominal Encoding : {result1}")
        ff = pd.DataFrame(result1, columns=one_hot_encoder.get_feature_names_out())
        x_test_cat.reset_index(drop=True, inplace=True)
        ff.reset_index(drop=True, inplace=True)
        x_test_cat = pd.concat([x_test_cat, ff], axis=1)
        x_test_cat = x_test_cat.drop(["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","PaperlessBilling","PaymentMethod"], axis=1)
        logger.info(f"Test Data Columns : \n {x_test_cat.columns}")

        # Ordinal Encoding
        logger.info(
            "-----------------Ordinal Encoding ------------------")
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(x_train_cat[["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"]])
        logger.info(f" features names : {ordinal_encoder.get_feature_names_out()}")
        result2 = ordinal_encoder.transform(x_train_cat[["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"]])
        fff = pd.DataFrame(result2, columns=ordinal_encoder.get_feature_names_out() + "_ordinal")
        x_train_cat.reset_index(drop=True, inplace=True)
        fff.reset_index(drop=True, inplace=True)
        x_train_cat = pd.concat([x_train_cat, fff], axis=1)
        x_train_cat = x_train_cat.drop(["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"], axis=1)
        logger.info(f" Train data columns : \n {x_train_cat.columns}")

        result3 = ordinal_encoder.transform(x_test_cat[["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"]])
        ffff = pd.DataFrame(result3, columns=ordinal_encoder.get_feature_names_out() + "_ordinal")
        x_test_cat.reset_index(drop=True, inplace=True)
        ffff.reset_index(drop=True, inplace=True)
        x_test_cat = pd.concat([x_test_cat, ffff], axis=1)
        x_test_cat = x_test_cat.drop(["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"], axis=1)
        logger.info(f" Test data columns : \n {x_test_cat.columns}")
        logger.info(f"no of columns of the training data{len(x_train_cat.columns)}")

        return x_train_cat, x_test_cat


    except Exception as e:
            logger.info(f"data loading failed")
            logger.error(e)
