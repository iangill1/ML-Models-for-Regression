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

#function to run gradient boosting regressor with hyperparameter tuning
def gbr_tuning_params(X_train, y_train, X_test):
    gbr = GradientBoostingRegressor()
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
    }
    #implement grid search
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    #get the best model from grid search
    best_gbr = grid_search.best_estimator_

    #get predictions on this model
    training_prediction = best_gbr.predict(X_train)
    test_prediction = best_gbr.predict(X_test)

    #return predictions and parameters that gave best results and best score
    #best score is mean cross-validated score of the best_estimator
    return training_prediction, test_prediction, grid_search.best_params_, grid_search.best_score_
