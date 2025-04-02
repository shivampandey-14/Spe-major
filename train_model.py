from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv("Assignment-2_Data.csv")

target=df['y']
train=df.drop('y',axis=1)
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=42)

for i in ['job','marital','month','poutcome','education','contact']:
  X_train = pd.get_dummies(X_train, columns=[i], drop_first=True)
  X_test = pd.get_dummies(X_test, columns=[i], drop_first=True)

X_train.info()

new=[]
for i in X_train['default']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_train['default']=new

new=[]
for i in X_train['housing']:
  if i=='yes':
    new.append(1)
  else:
    new.append(0)
X_train['housing']=new

new=[]
for i in X_train['loan']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_train['loan']=new

X_train.head()

X_train.info()

def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

weights, bias = initialize_weights(X_train.shape[1])

# Step 2: Train the model using gradient descent
learning_rate = 0.01
num_iterations = 10
def optimize(X, y, weights, bias, learning_rate, num_iterations):
    costs = []
    for i in range(num_iterations):
        # Compute predictions
        linear_model = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))  # Sigmoid function

        # Compute cost
        cost = -(1 / len(y)) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        costs.append(cost)

        # Compute gradients
        dw = (1 / len(y)) * np.dot(X.T, (predictions - y))
        db = (1 / len(y)) * np.sum(predictions - y)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, costs

weights, bias, costs = optimize(X_train.values, y_train.values, weights, bias, learning_rate, num_iterations)

# Step 3: Predict on new data
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = 1 / (1 + np.exp(-linear_model))  # Sigmoid function
    return [1 if p > 0.5 else 0 for p in predictions]

predictions = predict(X_test.values, weights, bias)


from sklearn.naive_bayes import GaussianNB
X_test = X_test[X_train.columns]

# Now fit and predict using the Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.prior = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


nb_classifier = NaiveBayes()


nb_classifier.fit(X_train.values, y_train.values)

y_pred_manual = nb_classifier.predict(X_test.values)

accuracy_manual = accuracy_score(y_test, y_pred_manual)
print(f"Accuracy of Naive Bayes implementation: {accuracy_manual}")

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
