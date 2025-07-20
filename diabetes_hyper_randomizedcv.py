from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model  import LogisticRegression
import numpy as np
import pandas as pd

diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop(["diabetes"], axis=1).values
y = diabetes_df["diabetes"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
kf = KFold(n_splits=5, shuffle=True, random_state=33)

#Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, num=50),
         "class_weight": ["balanced", {0: 0.8, 1: 0.2}]}

logreg = LogisticRegression()

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))