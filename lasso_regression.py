# Import Lasso
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt

sales_df = pd.read_csv("sales.csv")
X = sales_df.drop(["influencer","sales"], axis=1).values
y = sales_df["sales"].values
sales_columns = sales_df.drop(["influencer","sales"], axis=1).columns
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()