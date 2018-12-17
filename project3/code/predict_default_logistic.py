import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scikitplot as skplt

# Analysis of credit card data using logistic regression.

# Import data
## Temporarily extracted line 1 afrom dataset, the labels below.
labels = [
    "ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5",
    "BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4",
    "PAY_AMT5","PAY_AMT6","default payment next month"
]
path_to_data = "../data/"
filename = "default_of_credit_card_clients"
dataset = pd.read_csv(path_to_data+filename+".csv")
# Convert dtype to numeric
for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col])

# Split data into features and targets
x_full = dataset[dataset.columns[1:]].values
y_full = dataset[dataset.columns[-1]].values
# Split into training and test sets
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(
     x_full, y_full, test_size=test_size)
print("Dataset, x, y, shape: {}, {}, {}".format(
    dataset.shape, x_full.shape, y_full.shape))

# Limit number of training samples for speedy testing.
limit = int(len(x_train)*1.0)
# Run regression analysis
logistic = LogisticRegression(max_iter=100)
logistic.fit(x_train[:limit,:], y_train[:limit])
print("R2 score: ", logistic.score(x_test, y_test))
predicted_probabilities = logistic.predict_proba(x_test)

# Plot cumulative gain for comparison
def cumulative_gain_chart(targets, probabilities):
    probabilities = np.array([row[i] for row, i in zip(probabilities, targets)])
    vals = np.stack((targets, probabilities), axis=-1)
    n_defaults = len(targets[np.where(targets==1)])
    # Sort the stacked array and give reversed indices so it's in descending
    # order.
    vals_sorted = vals[vals[:,1].argsort()[::-1]]
    cumulative_sums = np.cumsum(vals_sorted[:,0])
    events = np.zeros(len(vals_sorted))
    events[:n_defaults] += 1
    cumulative_events = np.cumsum(events)
    plt.plot(range(len(vals_sorted)), cumulative_sums, label='Model')
    plt.plot(range(len(vals_sorted)), cumulative_events, label='Theory')
    print("Ratio non-defaults default 1 - d/total = ", 1 - n_defaults/len(targets))
    plt.show()




cumulative_gain_chart(y_test, predicted_probabilities)
#skplt.metrics.plot_confusion_matrix(y_test, predicted_probabilities)
#plt.show()
