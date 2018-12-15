import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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
print(dataset.columns[1:])
x_full = dataset[dataset.columns[1:]].values
y_full = dataset[dataset.columns[-1]].values
# Split into training and test sets
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(
     x_full, y_full, test_size=test_size, random_state=4)
print("Dataset, x, y, shape: {}, {}, {}".format(
    dataset.shape, x_full.shape, y_full.shape))

# Limit number of training samples for speedy testing.
limit = int(len(x_train)*0.1)
# Run network
net = MLPClassifier()
net.fit(x_train[:limit,:], y_train[:limit])
results = net.score(x_test, y_test)
print("Mean accuracy on test: ", results)
