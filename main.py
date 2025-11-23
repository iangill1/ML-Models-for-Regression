import pandas as pd
from sklearn.model_selection import KFold
from gradient_boosting import gbr_default_params, gbr_tuning_params

def main():
    #load dataset
    data = pd.read_csv("steel.csv")

    #specify target column and define features
    X = data.drop(columns=["tensile_strength"])
    y = data["tensile_strength"]

    #set up 10-fold cross validation with Kfold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    #calling functions and passing in data and KFold for cross validation
    default_results = gbr_default_params(X, y, kf)
    tuned_results = gbr_tuning_params(X, y, kf)

    #results
    print("Results with default params:\n", default_results)
    print("\n\nResults with tuned params:\n", tuned_results)

if __name__ == "__main__":
    main()