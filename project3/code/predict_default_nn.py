import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import scikitplot as skplt
from plotting_methods import cumulative_gain_chart
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler



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

sampling_types = ["none", "random_oversampling", "adasyn", "smote"]
if len(sys.argv) > 1 and sys.argv[1] in sampling_types:
    sampling_type = sys.argv[1]
else:
    print("Please provide one of the following resampling methods:")
    for s in sampling_types:
        print(s)
    exit(0)

# Resample data according to given sampling type
if sampling_type == "random_oversampling":
    x, y = RandomOverSampler().fit_resample(x_full, y_full)
elif sampling_type == "adasyn":
    x, y = ADASYN().fit_resample(x_full, y_full)
elif sampling_type == "smote":
    x, y = SMOTE().fit_resample(x_full, y_full)
else:
    x, y = x_full, y_full

# Split into training and test sets
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(
     x, y, test_size=test_size)
print("Dataset, x, y, shapes are: {}, {}, {}".format(
    dataset.shape, x.shape, y.shape))

# Run network
net = MLPClassifier(hidden_layer_sizes=(10,))
net.fit(x_train, y_train)
results = net.score(x_test, y_test)
print("Mean accuracy: ", results)

predicted_probabilities = net.predict_proba(x_test)
predictions = net.predict(x_test)

# Perform cross-validation
crossval_model = MLPClassifier(hidden_layer_sizes=(10,))
scores = cross_val_score(crossval_model, x, y, cv=5)
print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

filename = "../report/figures/net"
filename += "_" + sampling_type

# Set figsize for cumulative chart and confusion matrix and plot them
mpl.rcParams['figure.figsize'] = [4.0, 3.0]
cumulative_gain_chart(y_test, predicted_probabilities, filename)
plt.clf()

skplt.metrics.plot_confusion_matrix(y_test, predictions)
plt.tight_layout()
plt.savefig(filename + "_confmat.pdf", format="pdf")
plt.clf()

mpl.rcParams['figure.figsize'] = [5.0, 4.0]
skplt.metrics.plot_roc(y_test,
                       predicted_probabilities,
                       plot_micro=False,
                       plot_macro=False)
plt.tight_layout()
plt.savefig(filename + "_roc.pdf", format="pdf")
plt.clf()
