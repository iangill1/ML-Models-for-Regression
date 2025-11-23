from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#function to run gradient boosting regressor with default parameters and 10 fold cross validation
def gbr_default_params(X, y, kf):
    #make results reproducable with same random_state
    gbr = GradientBoostingRegressor(random_state=42)

    #cross validation with kfold and scoring for mse and r2
    cv_results = cross_validate(estimator=gbr,
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
        {
            #averages of these results for comparison with tuned results
            train_mse_avg,
            test_mse_avg,
            train_r2_avg,
            test_r2_avg
        }
    )

#function to run gradient boosting regressor with hyperparameter tuning
def gbr_tuning_params(X, y, kf):
    #making results reproducable with same random_state
    gbr = GradientBoostingRegressor(random_state=42)

    #define parameter grid for tuning
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.5],
        "n_estimators": [100, 200, 300, 400, 500, 600]
    }
    #implement grid search, swapping between mse and r2 for getting best model
    grid_search = GridSearchCV(estimator=gbr,
                               param_grid=param_grid,
                               cv=kf,
                               scoring=('neg_mean_squared_error', 'r2'),
                               refit='r2',
                               n_jobs=-1,
                               return_train_score=True)

    #fit the grid search to data
    grid_search.fit(X, y)
    #get the best model from grid search
    best_gbr = grid_search.best_estimator_
    #get the mean cross validated score from the best model
    best_score = grid_search.best_score_

    print("Best  GBR parameters found: ", grid_search.best_params_)
    print("Best GBR model cross-validated score: ", best_score)

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

    cv_results = pd.DataFrame(grid_search.cv_results_)
    heatmap_data = cv_results.pivot(
        index="param_learning_rate",
        columns="param_n_estimators",
        values="mean_test_neg_mean_squared_error"
    )
    heatmap_data = -heatmap_data  # Flip sign

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Grid Search MSE Heatmap")
    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")
    plt.show()

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
        test_r2_avg,
        grid_search
    )
