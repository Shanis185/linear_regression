
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm


from statsmodels.formula.api import ols


data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')


data.head()


data.shape


data[['TV', 'Radio', 'Social_Media']].describe()

# Calculate the average missing rate in the sales column.
missing_sales = data.Sales.isna().mean()

# Convert the missing_sales from a decimal to a percentage and round to 2 decimal places.
missing_sales = round(missing_sales * 100, 2)

# Display the results.
print('Percentage of promotions missing Sales: ' + str(missing_sales) + '%')

# Subset the data to include rows where Sales is present.
data = data.dropna(subset=['Sales'], axis=0)

# Create a histogram of the Sales.
fig = sns.histplot(data['Sales'])


fig.set_title('Distribution of Sales')

# Create a pairplot of the data.
sns.pairplot(data)


ols_formula = 'Sales ~ TV'


OLS = ols(formula=ols_formula, data=data)


model = OLS.fit()


model_results = model.summary()


print(model_results)

#scatterplot comparing X and Sales (Y).
sns.scatterplot(x=data['TV'], y=data['Sales'])

# Calculate the residuals.
residuals = model.resid

#1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#histogram with the residuals.
sns.histplot(residuals, ax=axes[0])


axes[0].set_xlabel("Residual Value")


axes[0].set_title("Histogram of Residuals")

#Q-Q plot of the residuals.
sm.qqplot(residuals, line='s', ax=axes[1])


axes[1].set_title("Normal Q-Q plot")


plt.tight_layout()


plt.show()

#scatterplot with the fitted values from the model and the residuals.
fig = sns.scatterplot(x=model.fittedvalues, y=model.resid)


fig.set_xlabel("Fitted Values")


fig.set_ylabel("Residuals")


fig.set_title("Fitted Values v. Residuals")


fig.axhline(0)


plt.show()


