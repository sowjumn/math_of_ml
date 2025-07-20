"""
Description: Hyperparameter tuning using GridSearchCV and LassoRegression 
Author: Naga Sowjanya Sutherland

"""
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd


#Get the Data
diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop(["diabetes"], axis=1).values
y = diabetes_df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=32)
# Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, num=20)}

# Instantiate lasso_cv and Kfold 
lasso = Lasso()
kf = KFold(n_splits=5, shuffle=True, random_state=31)


# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)
lasso_cv.fit(X_train,y_train)

print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))