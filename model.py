import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
# Load the csv file
df = pd.read_csv("final_test_1.csv")

print(df.head())

# Select independent and dependent variable
X = df[['Time_under90(Min)', 'Snoring Size(1:mild,2:morderate,3:severe)', 'SpO2_AVG', 'SpO2_MIN', 'BMI' ]]
y = df['AHI']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperopt objective function for BaggingRegressor with 10-fold cross-validation
def objective_bagging(params):
    base_estimator = LGBMRegressor(**params, random_state=42)
    bagging_model = BaggingRegressor(base_estimator=base_estimator, n_estimators=5, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Increased the number of folds to 10
    cv_scores = cross_val_score(bagging_model, X_train_scaled, y_train, cv=kf, scoring='r2')
    mean_cv_r2 = np.mean(cv_scores)

    # As hyperopt minimizes the objective function, we negate the mean CV R-squared
    return -mean_cv_r2

# Hyperopt search space for BaggingRegressor with an extended range
space_bagging = {
    'n_estimators': hp.choice('n_estimators', range(50, 501)),  # Extended the range to 50-500
    'max_depth': hp.choice('max_depth', range(1, 31)),  # Extended the range to 1-30
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),  # Extended the range
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'min_child_samples': hp.choice('min_child_samples', range(1, 21)),  # Extended the range to 1-20
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(1.0))
}

# Hyperopt optimization for BaggingRegressor
best_bagging = fmin(fn=objective_bagging, space=space_bagging, algo=tpe.suggest, max_evals=50)

best_params_bagging = {
    'n_estimators': best_bagging['n_estimators'] + 50,
    'max_depth': best_bagging['max_depth'] + 1,
    'learning_rate': best_bagging['learning_rate'],
    'subsample': best_bagging['subsample'],
    'colsample_bytree': best_bagging['colsample_bytree'],
    'min_child_samples': best_bagging['min_child_samples'] + 1,
    'reg_alpha': best_bagging['reg_alpha'],
    'reg_lambda': best_bagging['reg_lambda']
}

# Create the best BaggingRegressor model
best_base_estimator = XGBRegressor(**best_params_bagging, random_state=42)
best_bagging_model = BaggingRegressor(base_estimator=best_base_estimator, n_estimators=5, random_state=42)

# Fit the model on the entire training data
best_bagging_model.fit(X_train_scaled, y_train)

# Make pickle file of our model
pickle.dump(best_bagging_model, open("ahi_model_xgb.pkl", "wb"))