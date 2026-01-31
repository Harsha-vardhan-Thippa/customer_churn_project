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
logger = setup_logging("variable_transformation")

from scipy import stats

def Variable_Transformation(x_train_num,x_test_num):
    try:
        logger.info(f"{x_train_num.columns}---->{x_train_num.shape}")
        logger.info(f"{x_test_num.columns}---->{x_test_num.shape}")

        # for i in x_train_num.columns:
        #      sns.boxplot(x=x_train_num[i])
        #      plt.show()

        for i in x_train_num.columns:
            x_train_num[i + "_yeo"], lam = stats.yeojohnson(x_train_num[i])
            x_train_num = x_train_num.drop(i, axis=1)
            iqr = x_train_num[i + "_yeo"].quantile(0.75) - x_train_num[i + "_yeo"].quantile(0.25)
            upperlimit = x_train_num[i + "_yeo"].quantile(0.75) + (1.5 * iqr)
            lowerlimit = x_train_num[i + "_yeo"].quantile(0.25) - (1.5 * iqr)

            x_train_num[i + "_yeo_trim"] = np.where(x_train_num[i + "_yeo"] > upperlimit, upperlimit,
                                                    np.where(x_train_num[i + "_yeo"] < lowerlimit, lowerlimit,
                                                             x_train_num[i + "_yeo"]))
            x_train_num = x_train_num.drop([i + "_yeo"], axis=1)

            x_test_num[i + "_yeo_trim"] = np.where(x_test_num[i] > upperlimit, upperlimit,
                                                   np.where(x_test_num[i] < lowerlimit, lowerlimit, x_test_num[i]))
            x_test_num = x_test_num.drop(i, axis=1)
        logger.info(f"{x_train_num.columns}---->{x_train_num.shape}")
        logger.info(f"{x_test_num.columns}---->{x_test_num.shape}")

        # for i in x_train_num.columns:
        #      sns.boxplot(x=x_train_num[i])
        #      plt.show()
        return x_train_num, x_test_num


    except Exception as e:
        logger.info("data failed loading")
        logger.error(e)
