import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
data = pd.read_csv("marketing_sales_data.csv")


data = data.dropna(axis=0)
##frst plotting 
sns.pairplot(data)

# Select relevant columns
# Save resulting DataFrame in a separate variable to prepare for regression



ols_data = data[["Radio", "Sales"]]

# Write the linear regression formula
# Save it in a variable



ols_formula = "Sales ~ Radio"

# Implement Ols

OLS = ols(formula = ols_formula, data = ols_data)
# Fit the model to the data
# Save the fitted model in a variable
model = OLS.fit()
model.summary()


# Plot the OLS data with the best fit regression line

sns.regplot(x = "Radio", y = "Sales", data = ols_data)

#---------------------------------------------------------
#checking for normality
residuals = model.resid

fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()

##qq plot for confirmation of normality
sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()

fitted_values = model.predict(ols_data["Radio"])
# scatterplot of residuals against fitted values
fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()