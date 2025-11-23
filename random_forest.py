from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate

#function to run random forest regressor with default parameters and 10 fold cross validation
def rf_default_params(X, y, kf):
    #make results reproducable with same random_state
    rf = RandomForestRegressor(random_state=42)

    #cross validation with kfold and scoring for mse and r2
    cv_results = cross_validate(estimator=rf,
                                X=X,
                                y=y,
                                cv=kf,
                                scoring=("neg_mean_squared_error", "r2"),
                                return_train_score=True)

    #get results from cross validation, converting negative mse to positive
    cv_train_mse_error = (-cv_results['train_neg_mean_squared_error']).tolist()
    cv_test_mse_error = (-cv_results['test_neg_mean_squared_error']).tolist()
    cv_train_r2_error = cv_results['train_r2'].tolist()
    cv_test_r2_error = cv_results['test_r2'].tolist()

    #get average of results across all folds
    train_mse_avg = np.mean(cv_train_mse_error)
    test_mse_avg = np.mean(cv_test_mse_error)
    train_r2_avg = np.mean(cv_train_r2_error)
    test_r2_avg = np.mean(cv_test_r2_error)

    return (
        {
            #all results from cross validation
            "cv_train_mse_error": cv_train_mse_error,
            "cv_test_mse_error": cv_test_mse_error,
            "cv_train_r2_error": cv_train_r2_error,
            "cv_test_r2_error": cv_test_r2_error,
        },
        #averages of these results for comparison with tuned results
        train_mse_avg,
        test_mse_avg,
        train_r2_avg,
        test_r2_avg
    )

#function to run random fprest regressor with hyperparameter tuning
def rf_tuning_params(X, y, kf):
    #making results reproducable with same random_state
    rf = RandomForestRegressor(random_state=42)

    #define parameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500, 600],
        "max_depth": [None, 10, 20, 30, 40, 50]
    }
    #implement grid search, swapping between mse and r2 for getting best model
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=kf,
                               scoring=('neg_mean_squared_error', 'r2'),
                               refit='neg_mean_squared_error',
                               n_jobs=-1)

    #fit the grid search to data
    grid_search.fit(X, y)
    #get the best model from grid search
    best_gbr = grid_search.best_estimator_
    #get the mean cross validated score from the best model
    best_score = grid_search.best_score_

    print("Best RF parameters found: ", grid_search.best_params_)
    print("Best RF model cross-validated score: ", best_score)

    #cross validation with kfold and scoring for mse and r2 using best model
    cv_results = cross_validate(estimator=best_gbr,
                                X=X,
                                y=y,
                                cv=kf,
                                scoring=("neg_mean_squared_error", "r2"),
                                return_train_score=True)

    #get results from cross validation, converting negative mse to positive
    cv_train_mse_error = (-cv_results['train_neg_mean_squared_error']).tolist()
    cv_test_mse_error = (-cv_results['test_neg_mean_squared_error']).tolist()
    cv_train_r2_error = cv_results['train_r2'].tolist()
    cv_test_r2_error = cv_results['test_r2'].tolist()

    # get average of results across all folds
    train_mse_avg = np.mean(cv_train_mse_error)
    test_mse_avg = np.mean(cv_test_mse_error)
    train_r2_avg = np.mean(cv_train_r2_error)
    test_r2_avg = np.mean(cv_test_r2_error)

    return (
        {
            #all results from cross validation
            "cv_train_mse_error": cv_train_mse_error,
            "cv_test_mse_error": cv_test_mse_error,
            "cv_train_r2_error": cv_train_r2_error,
            "cv_test_r2_error": cv_test_r2_error,
        },
        #averages of these results for comparison with tuned results
        train_mse_avg,
        test_mse_avg,
        train_r2_avg,
        test_r2_avg
    )

