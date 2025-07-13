from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=5)
#Input variables
X = churn_df[["account_length", "customer_service_calls"]].values
#target variables
y = churn_df["churn"].values

knn.fit(X,y)

X_new =  np.array([[56.8, 17.5],
                   [24.4, 24.1],
                   [50.1, 10.9]])

# 3 observations and 2 features
print(X_new.shape)

predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))