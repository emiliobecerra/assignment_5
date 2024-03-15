import pandas
import numpy

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import classification_report

import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML, display

import seaborn as sns
import matplotlib.pyplot as plt

catalog_data = pandas.read_csv("catalog.csv")

estimation_sample = catalog_data[catalog_data['holdout'] == 0]
holdout_sample = catalog_data[catalog_data['holdout'] == 1]

estimation_sample.to_csv("estimation_sample.csv", index=False)
holdout_sample.to_csv("holdout_sample.csv", index=False)

estimation_sample = pandas.read_csv("estimation_sample.csv")


target = estimation_sample.iloc[:,1]
data = estimation_sample.iloc[:,2:9]


#Let's make a machine to train
machine = linear_model.LogisticRegression()
machine.fit(data,target)

#Let's look at the results
independent_vars = ['tabordrs', 'divsords', 'divwords',
 'spgtabord', 'moslsdvs', 'moslsdvw', 'moslstab', 'orders']
estimation_sample['const'] = 1
dependent_var = 'buytabw'
logit_model = sm.Logit(estimation_sample[dependent_var],
 estimation_sample[independent_vars + ['const']])
result = logit_model.fit()

stargazer = Stargazer([result])
with open('regression_table.tex', 'w') as f:
    f.write(stargazer.render_latex())

#Let's get prediction probabilities
holdout_sample = pandas.read_csv("holdout_sample.csv")

independent_vars = ['tabordrs', 'divsords', 'divwords',
 'spgtabord', 'moslsdvs', 'moslsdvw', 'moslstab', 'orders']
holdout_sample['const'] = 1
dependent_var = 'buytabw'
holdout_res = sm.Logit(holdout_sample[dependent_var],
 holdout_sample[independent_vars + ['const']])
result = holdout_res.fit()
holdout_sample['Predicted_Probability'] = result.predict(holdout_sample[independent_vars
 + ['const']])
print(holdout_sample[['buytabw', 'Predicted_Probability']])

#Let's Predict

new_data = holdout_sample.iloc[:,2:9]
prediction = machine.predict(new_data)
print(prediction)
holdout_sample['Prediction'] = prediction 

holdout_sample.to_csv("holdout_sample_with_prediction.csv", index=False)


#Let's Validate
holdout_with_prediction = pandas.read_csv('holdout_sample_with_prediction.csv')
selected_columns = holdout_with_prediction[['buytabw', 'Prediction']]

print(selected_columns)

target_test = holdout_sample.iloc[:,1]

confusion_matrix = metrics.confusion_matrix(target_test, prediction)
print("Confusion matrix: ")
print(confusion_matrix)
print("\n\n")

#Let's create boxplots
holdout_sample = pandas.read_csv('holdout_sample_with_prediction.csv')

plt.figure(figsize=(10, 6))
sns.boxplot(x='buytabw', y='Predicted_Probability', data=holdout_sample)
means = holdout_sample.groupby('buytabw')['Predicted_Probability'].mean()
plt.xlabel('Actual Purchase Decision')
plt.ylabel('Predicted Purchase Probability')
plt.title('Box plot of Actual Purchase Decision vs. Predicted Purchase Probability')
plt.show()

#Let's rank probabilities through deciles
holdout_sample.sort_values(by='Predicted_Probability',
 ascending=True, inplace=True)
holdout_sample['rank'] = range(1, len(holdout_sample) + 1)

num_deciles = 10
decile_size = len(holdout_sample) // num_deciles
holdout_sample['group'] = pandas.cut(holdout_sample['rank'],
 bins=num_deciles, labels=range(1, num_deciles + 1))

plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='Predicted_Probability',
 data=holdout_sample, order=range(1, num_deciles + 1))
plt.xlabel('Group (Deciles)')
plt.ylabel('Predicted Purchase Probability')
plt.title('Box plot of Group (Deciles) vs. Predicted Purchase Probability')
plt.show()

holdout_sample.to_csv("holdout_sample_with_prediction_and_rank.csv", index=False)

#Let's create a marketing strategy
holdout_sample = pandas.read_csv('holdout_sample_with_prediction_and_rank.csv')
cumulative_observations = holdout_sample.groupby('group')['rank'].count().cumsum()
cumulative_buyers = holdout_sample.groupby('group')['buytabw'].sum().cumsum()
holdout_sample['Cumulative_Observations'] = holdout_sample.groupby('group')['rank'].cumcount() + 1
holdout_sample['Cumulative_Buyers'] = holdout_sample.groupby('group')['buytabw'].cumsum()

cumulative_percentage_mailed = cumulative_observations / len(holdout_sample) * 100
percentage_buyers_captured = cumulative_buyers / holdout_sample['buytabw'].sum() * 100

plt.figure(figsize=(10, 6))
plt.plot(cumulative_percentage_mailed, percentage_buyers_captured, marker='o', linestyle='-')
plt.xlabel('Cumulative % of Mailed Tabloids')
plt.ylabel('% of Buyers Captured by Mailings')
plt.title('Gains Chart')
plt.grid(True)
plt.show()

print(holdout_sample[['group',
 'rank', 
 'buytabw', 
 'Cumulative_Observations', 
 'Cumulative_Buyers']])

#Let's calculate estimated profits
average_margin_per_customer = 19.5  
cost_of_printing_and_mailing = 1.0  

holdout_sample['Expected_Profit'] = holdout_sample['Predicted_Probability'] * average_margin_per_customer - cost_of_printing_and_mailing

# Plot histogram of profit variable
plt.figure(figsize=(10, 6))
plt.hist(holdout_sample['Expected_Profit'],
 bins=30, edgecolor='black')
plt.xlabel('Expected Profit in Dollars')
plt.ylabel('Frequency')
plt.title('Histogram of Expected Profit per Customer')
plt.grid(True)
plt.show()

# Calculate fraction of customers that are profitable in expectation
fraction_profitable = (holdout_sample['Expected_Profit'] > 0).mean()
print(f"Fraction of customers that are profitable in expectation: {fraction_profitable:.2f}")

#Let's make a mailing decision
num_buyers = (holdout_sample['buytabw'] == 1).sum()
actual_profits = num_buyers * (average_margin_per_customer -
 cost_of_printing_and_mailing)
print(f"Actual profits from customers who bought: ${actual_profits:.2f}")

total_observations = len(holdout_sample)
num_profitable_observations = int(total_observations * fraction_profitable)
expected_profits_profitable_customers = num_profitable_observations * (average_margin_per_customer - cost_of_printing_and_mailing)
print(f"Expected profits if we only mailed to profitable customers: ${expected_profits_profitable_customers:.2f}")

blanket_profits = holdout_sample['Expected_Profit'].sum()
print(f"Expected profits if we mailed to every customer: ${blanket_profits:.2f}")

#It's better if we only mail to people who are in the predicted proftiable fraction group. 
Percent_Improvement = ((expected_profits_profitable_customers - blanket_profits) /expected_profits_profitable_customers) * 100
print(f"Percentage Change Improvement going from Blanket Mailing to Profitable Customer Only Mailing: {Percent_Improvement:.2f}%")

holdout_sample.to_csv("Final_Holdout_Sample_Data.csv", index=False)