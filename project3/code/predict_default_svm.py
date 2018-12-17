import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
x_full = dataset[dataset.columns[1:]].values
y_full = dataset[dataset.columns[-1]].values
# Split into training and test sets
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(
     x_full, y_full, test_size=test_size, random_state=2)
print("Dataset, x, y, shapes are: {}, {}, {}".format(
    dataset.shape, x_full.shape, y_full.shape))

limit = int(len(y_train)*1.0)

svc = SVC()
svc.fit(x_train[:limit, :], y_train[:limit])
print(svc.score(x_test, y_test))
