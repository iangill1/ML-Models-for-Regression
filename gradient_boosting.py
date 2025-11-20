from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import GridSearchCV

#function to run gradient boosting regressor with default parameters
def gbr_default_params(X_train, y_train, X_test):
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)

    #get predictions of target variable
    training_prediction = gbr.predict(X_train)
    test_prediction = gbr.predict(X_test)

    return training_prediction, test_prediction
