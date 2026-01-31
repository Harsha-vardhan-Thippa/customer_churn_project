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
logger = setup_logging("Feature_selection")

from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold

reg_con = VarianceThreshold(threshold = 0.0)
reg_quesi = VarianceThreshold(threshold = 0.1)

def complete_feature_selection(x_train_num,x_test_num,y_train):
    try:
        logger.info(f"{x_train_num.columns}---->{x_train_num.shape}")
        logger.info(f"{x_test_num.columns}---->{x_test_num.shape}")
        # constant technique
        reg_con.fit(x_train_num)
        logger.info(f"columns to remove (constant)(un used columns): {x_train_num.columns[~reg_con.get_support()]}")
        good_data = reg_con.transform(x_train_num)
        good_data1 = reg_con.transform(x_test_num)
        x_train_fs = pd.DataFrame(data=good_data, columns=x_train_num.columns[reg_con.get_support()])
        x_test_fs = pd.DataFrame(data=good_data1, columns=x_test_num.columns[reg_con.get_support()])
        # quesi constant technique
        reg_quesi.fit(x_train_fs)
        logger.info(
            f"columns to remove (quesi constant)(un used columns): {x_train_fs.columns[~reg_quesi.get_support()]}")
        good_data2 = reg_quesi.transform(x_train_fs)
        good_data3 = reg_quesi.transform(x_test_fs)
        x_train_fs_1 = pd.DataFrame(data=good_data2, columns=x_train_fs.columns[reg_quesi.get_support()])
        x_test_fs_2 = pd.DataFrame(data=good_data3, columns=x_test_fs.columns[reg_quesi.get_support()])
        logger.info(f"{x_train_fs_1.columns}---->{x_train_fs_1.shape}")
        logger.info(f"{x_test_fs_2.columns}---->{x_test_fs_2.shape}")
        logger.info("=====================================================================")
        logger.info(f"{y_train.unique()}")
        # Hypothesis with co relation
        y_train = y_train.map({"Yes": 1, "No": 0}).astype(int)
        values = []
        for i in x_train_fs_1.columns:
            values.append(pearsonr(x_train_fs_1[i], y_train))
        values = np.array(values)
        # print(values)
        plt.figure(figsize=(10, 10))
        p_values = pd.Series(values[:, 1], index=x_train_fs_1.columns)
        p_values.sort_values(ascending=False, inplace=True)
        # print(p_values)
        logger.info(f"total columns:{p_values}")
        # logger.info(f"column to remove : {p_values[0]}")
        # p_values.plot(kind = 'bar', figsize = (10,4),)
        # plt.show()
        x_train_fs_1 = x_train_fs_1.drop(["MonthlyCharges_yeo_trim"], axis=1)
        x_test_fs_2 = x_test_fs_2.drop(["MonthlyCharges_yeo_trim"], axis=1)
        logger.info(f"After completing the hypothesis co relation ")
        logger.info(f"{x_train_fs_1.columns}---->{x_train_fs_1.shape}")
        logger.info(f"{x_test_fs_2.columns}---->{x_test_fs_2.shape}")

        return x_train_fs_1, x_test_fs_2


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        logger.error("Exception occurred")
        logger.error(f"Type    : {exc_type.__name__}")
        logger.error(f"Message : {e}")
        logger.error(f"File    : {file_name}")
        logger.error(f"Line    : {line_no}")



