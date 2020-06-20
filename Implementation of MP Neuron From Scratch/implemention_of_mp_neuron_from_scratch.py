"""

	Implemention of MP Neuron From Scratch Using Numpy.


* MP Neuron is the first mathematical model of a neural network proposed by Warren McCulloch and Walter Pitts in 1943
* Inputs can be only binary values either 0 or 1
* Outputs can be 0 or 1
* MP Neuron can only used for classification.

#### Source:
* http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html
* http://www.mind.ilstu.edu/curriculum/modOverview.php?modGUI=212

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the data
breast_cancer = load_breast_cancer()

# Get X and y pair
X,y = breast_cancer.data, breast_cancer.target
X.shape,y.shape

# Create the dataframe from the X and y pair
df = pd.DataFrame(X,columns=breast_cancer.feature_names)
df['class'] = breast_cancer.target

df.head()

# shape of dataframe
df.shape

# Get the info of dataframe about columns datatype,count of null values and memory usage
df.info()

# Get Summary statistics of numerical features
df.describe()

df.groupby('class').describe()

df.groupby('class').mean()

# Get the class distribution
df['class'].value_counts()

y = df['class'].value_counts()
x = df['class'].value_counts().index
plt.bar(x,y,color='orange')
plt.xlabel('classes')
plt.ylabel('count')
plt.title('Distribution of target')
plt.xticks([0,1])
plt.show()

X = df.drop('class',axis=1)
y = df['class']
X.shape, y.shape

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

print('Shape of X_train:{} and y_train:{}'.format(X_train.shape,y_train.shape))
print('Shape of X_test:{} and y_test:{}'.format(X_test.shape,y_test.shape))

# check the distribution of target in train and test split
# fig, ax = plt.subplots(1,2)
fig, ax = plt.subplots(1,2,figsize=(8,4))

y1 = y_train.value_counts(normalize=True)
x1 = list(y_train.value_counts().index)
ax[0].bar(x1,y1,color='orange')
ax[0].set_ylabel('count')
ax[0].set_xlabel('Train classes')
ax[0].set_xticks([0,1])

y2 = y_test.value_counts(normalize=True)
x2 = y_test.value_counts().index
ax[1].bar(x2,y2,color='orange')
ax[1].set_xlabel('Test classes')
ax[1].set_xticks([0,1])

plt.show()

# plotting train data to see how data is distributed across features
plt.figure(figsize=(12,6))
plt.plot(X_train.T,"*")
plt.xticks(rotation='90')
plt.show()

# MP Neuron requires features should be binary not continous
X_train_bin = X_train.apply(pd.cut,bins=2, labels=[1,0])
X_test_bin = X_test.apply(pd.cut,bins=2, labels=[1,0])

# plot binarized train and test data to see how data is distributed across features
fig, ax = plt.subplots(1,2,figsize=(16,3.5))

ax[0].plot(X_train_bin.T,"*")
ax[0].set_xticklabels(labels=breast_cancer.feature_names,rotation='90')

ax[1].plot(X_test_bin.T,"*")
ax[1].set_xticklabels(labels=breast_cancer.feature_names,rotation='90')

plt.show()

# Convert train and test dataframes to numpy
X_train_binarised = X_train_bin.values
X_test_binarised = X_test_bin.values

class MPNeuron:
  def __init__(self):
    self.b = 0
  
  def train(self, X, Y):
    b_values = range(X.shape[1]+1)
    best_acc = 0
    for val in b_values:
      acc = 0
      for x,y in zip(X,Y):
        y_pred = np.sum(x) >= val
        if (y==y_pred):
          acc+=1
      if acc > best_acc:
        best_acc = acc
        self.b = val
    return self
  
  def predict(self,X):
    preds = []
    for x in X:
      y_pred = np.sum(x) >= self.b
      preds.append(y_pred)
    return preds

# Create object of MPNeuron
model = MPNeuron()
model.train(X_train_binarised,y_train)
y_pred_train = model.predict(X_train_binarised)
y_pred_test = model.predict(X_test_binarised)

from sklearn.metrics import accuracy_score, f1_score

print('Train Accuracy:{} and F1_Score:{}'.format(accuracy_score(y_train,y_pred_train), f1_score(y_train,y_pred_train)))
print('Test Accuracy:{} and F1_Score:{}'.format(accuracy_score(y_test,y_pred_test), f1_score(y_test,y_pred_test)))

print('Threshold value b:',model.b)

"""### References

* https://matplotlib.org/3.1.0/api/axes_api.html#matplotlib.axes.Axes
"""