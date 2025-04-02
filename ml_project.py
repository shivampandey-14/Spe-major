import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/app/data/dataset.csv")

df.shape

df.describe()


df.sample(10)

df.info()

df.isnull().sum()

df.duplicated().sum()

df['y'].value_counts()

df['age'].unique()

# df['y']=np.where(df['y']=='yes',1,0)
y_new=[]
for i in df['y']:
  if i=="yes":
    y_new.append(1)
  else:
    y_new.append(0)
df['y']=y_new
df['y'].value_counts()

df.corr(numeric_only=True)

df.sample(5)

df['y'].value_counts().plot(kind='bar') #to visualize the categorical data we can use pie char or bar graph

# to analyze the numerical data in the given data set we can plot histogram.

import matplotlib.pyplot as plt

plt.hist(df['age'],bins=100)

sns.distplot(df['age'],bins=100)

# sns.boxplot(df['age'])

df.boxplot(column='age')

# df['age']=np.where(df['age']>100,df['age'].median(),df['age'])

age_list = df['age'].tolist()
median_age = sorted(age_list)[len(age_list) // 2]
new_age_list = []
for age in age_list:
  if age > 100:
    new_age_list.append(median_age)
  else:
    new_age_list.append(age)
df['age'] = new_age_list


Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for non-outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter the DataFrame to keep only non-outliers
df = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

df.boxplot(column='age')

df.boxplot(column='balance')

Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for non-outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter the DataFrame to keep only non-outliers
df = df[(df['balance'] >= lower_bound) & (df['balance'] <= upper_bound)]

sns.boxplot(df['balance'])

df.info()

# df['job'].value_counts()

job_counts = {}
for job in df['job']:
  if job in job_counts:
    job_counts[job] += 1
  else:
    job_counts[job] = 1

for job, count in job_counts.items():
  print(f"{job}: {count}")


df['job'].value_counts().plot(kind='bar', figsize=(12,6));

# We can see that most of the clients beloned to
# blue-collar job are most  and students are least in general as they don't make term deposits in general.

# prompt: maratial status

df['marital'].value_counts()
df['marital'].value_counts().plot(kind='bar', figsize=(12,6));


marital_y_counts = df.groupby(['marital', 'y'])['y'].count().unstack()
marital_y_counts.plot(kind='bar', figsize=(12,6))
plt.title('Marital Status vs. Subscription Count')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(['No Subscription', 'Subscription'])
plt.show()


# bivariate analysis
# Numeric-Numeric data corelation
# scaterplot
# palirplot
# line plot
# Numeric-Categorical
# barplot
# boxplot
# distplot
# Categorical-Categorical
# heatmap
# clustermap

print(pd.crosstab(df['job'],df['y']))
sns.scatterplot(x=df['age'],y=df['y'])
# no relation is found


pd.crosstab(df['job'],df['y'])
plt.figure(figsize=(14, 6))  # Set figsize here
ax = plt.gca()
sns.barplot(x=df['job'],y=df['y'],ax=ax)
# From the above graph we can infer that students and retired people have higher chances of subscribing to a term deposit,
# which is surprising as students generally do not subscribe to a term deposit.
# The possible reason is that the number of students in the dataset
# is less and comparatively to other job types, more students have subscribed to a term deposit.

pd.crosstab(df['marital'],df['y'])
plt.figure(figsize=(14, 6))  # Set figsize here
ax = plt.gca()
sns.barplot(x=df['marital'],y=df['y'],ax=ax)
# here it is not clear who is taking the term deposite as all of them are closer to each other.

sns.scatterplot(x=df['balance'],y=df['y'])
# Here we can see that only those poeople who have >-1000 balance are subscribing for term deposite.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlation(df):

  numeric_cols = df.select_dtypes(include=np.number).columns
  correlation_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)

  for col1 in numeric_cols:
    for col2 in numeric_cols:
      if col1 == col2:
        correlation_matrix.loc[col1, col2] = 1.0
      else:
        # Use .values to get the underlying NumPy array, ensuring numeric indexing
        correlation_matrix.loc[col1, col2] = calculate_pearson_correlation(df[col1].values, df[col2].values)

  # Convert all values in the correlation matrix to numeric
  correlation_matrix = correlation_matrix.astype(float)
  return correlation_matrix

def calculate_pearson_correlation(series1, series2):
  """
  Calculates the Pearson correlation coefficient between two series.

  Args:
    series1: A pandas Series or NumPy array.
    series2: A pandas Series or NumPy array.

  Returns:
    The Pearson correlation coefficient.
  """
  n = len(series1)
  if n != len(series2):
    raise ValueError("Series must have the same length")

  mean1 = sum(series1) / n
  mean2 = sum(series2) / n

  # Use numeric indexing to avoid KeyError
  numerator = sum([(series1[i] - mean1) * (series2[i] - mean2) for i in range(n)])
  denominator = ((sum([(series1[i] - mean1) ** 2 for i in range(n)])) ** 0.5) * \
                ((sum([(series2[i] - mean2) ** 2 for i in range(n)])) ** 0.5)

  if denominator == 0:
    return 0  # Handle the case of zero variance
  else:
    return numerator / denominator

correlation_matrix_manual = calculate_correlation(df)
print(correlation_matrix_manual)
sns.heatmap(correlation_matrix_manual,annot=True,cmap='coolwarm') # Now this line should work

# We can infer that duration of the call is highly correlated with the target variable. As the duration of the call is more,
# there are higher chances that the client is showing interest in the term deposit and
# hence there are higher chances that the client will subscribe to term deposit.

# Model Building


from sklearn.model_selection import train_test_split


target=df['y']
train=df.drop('y',axis=1)
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=42)


def one_hot_encode(df, column_name):
  unique_values = df[column_name].unique()
  for value in unique_values:
    df[column_name + "_" + str(value)] = (df[column_name] == value).astype(int)
  df.drop(column_name, axis=1,inplace=True)


for i in ['job','marital','month','poutcome','education','contact']:
  one_hot_encode(X_train,i)
  one_hot_encode(X_test,i)

X_train.info()

sns.barplot(x=df['default'],y=df['y'])

new=[]
for i in X_train['default']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_train['default']=new

new=[]
for i in X_test['default']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_test['default']=new

new=[]
for i in X_train['housing']:
  if i=='yes':
    new.append(1)
  else:
    new.append(0)
X_train['housing']=new

new=[]
for i in X_test['housing']:
  if i=='yes':
    new.append(1)
  else:
    new.append(0)

X_test['housing']=new

new=[]
for i in X_train['loan']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_train['loan']=new

new=[]
for i in X_test['loan']:
  if i=='yes':
    new.append(0)
  else:
    new.append(1)
X_test['loan']=new

X_train.head()

X_train.info()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias
def compute_cost(y, W, X, b):
    m = len(y)
    linear_model = np.dot(X, W) + b
    cost = (1 / m) * np.sum(np.log(1 + np.exp(-y * linear_model)))
    return cost

def compute_gradients(X, y, W, b):
    m = X.shape[0]
    linear_model = np.dot(X, W) + b
    z = -y * linear_model
    sigmoid_z = sigmoid(z)
    dw = -np.dot(X.T, y * sigmoid_z)
    db = -np.mean(y * sigmoid_z)
    return dw, db


def optimize(X, y, weights, bias, learning_rate, num_iterations):
    costs = []
    for i in range(num_iterations):
        cost = compute_cost(y, weights, X, bias)
        costs.append(cost)
        dw, db = compute_gradients(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return weights, bias, costs

def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    predictions = np.where(y_pred >= 0.5, 0, 1)
    return predictions


weights, bias = initialize_weights(X_train.shape[1])

# Step 2: Train the model using gradient descent
learning_rate = 0.01
num_iterations = 10
weights, bias, costs = optimize(X_train, y_train, weights, bias, learning_rate, num_iterations)

# Step 3: Predict on new data
predictions = predict(X_test, weights, bias)


from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)

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

accuracy_score(y_test,y_pred_dt)
