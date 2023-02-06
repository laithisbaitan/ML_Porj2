import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn import svm
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

# Load the data
df = pd.read_excel('WeatherData.xls')

# Count of missing values
missing_values = df.isna().sum()
print(missing_values)
# looking on the missing values we can see that we can't remove them because they will affect the data (data loss)

# filling the data using median
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        median = df[column].median()
        df[column].fillna(median, inplace=True)

# filling the missing category values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
outliers = pd.DataFrame
# calculating the outliers for each column
for column in df.columns:
    if df[column].dtype == "float64":
        Q1 = float(df[column].quantile(0.25))
        Q3 = float(df[column].quantile(0.75))

        IQR = float(Q3 - Q1)
        n1 = Q1 - (1.4 * IQR)
        n2 = Q3 + (1.4 * IQR)
        outliers = df[(df[column] < n1) | (df[column] > n2)]

        df[column].loc[outliers.index] = df[column].median()
        # df = df[~df.index.isin(outliers.index)]

# box plot for outliers

# fig2 = plt.figure(figsize=(10, 10))
# i=0
# for region in range(1, 5):
#     i += 1
#     ax = fig2.add_subplot(1, 4, i )
#     regionp = df[df['Location'] == 'Region'+str(region)]
#     ax.boxplot(regionp['MaxTemp'])
#     ax.set_title('Region'+str(region))
#     tick_positions = np.arange(start=min(df['MaxTemp']), stop=max(df['MaxTemp']), step=2)
#     plt.yticks(tick_positions)
# plt.subplots_adjust(wspace=2, hspace=2)
# plt.show()
#

# converting categorical values using get dummies
df = df.drop(columns=['Date'], axis=1)
df = df.drop(columns=['Location'], axis=1)

df = pd.get_dummies(df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])
df['RainToday2'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow2'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

df = df.drop(columns=['RainTomorrow'], axis=1)
df = df.drop(columns=['RainToday'], axis=1)

# corr_matrix = df.corr()
# fig, ax = plt.subplots(figsize=(20, 18))
# sns.heatmap(corr_matrix, ax=ax)
# sns.heatmap(corr_matrix, ax=ax, annot=True , fmt='.1g')
# plt.show()

# spliting data into 80% training and 20% testing 
Y = df['RainTomorrow2']
X = df.drop('RainTomorrow2', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# *************Naive Bayes***************
def NB():
    print("******Naive Bayes********")
    # create the naive bayes model
    NBmodel = GaussianNB()

    # fit the model on the training data
    NBmodel.fit(X_train, y_train)

    # make predictions on the test data
    y_pred = NBmodel.predict(X_test)

    # evaluate the model performance

    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    y_scores = NBmodel.predict_proba(X_test)[:, 1]
    # calculate AUC
    print('AUC:', roc_auc_score(y_test, y_scores))

    # generate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(matrix)


# *************KNN***************
def KNN():
    print("******KNN********")
    # Odd or Even
    neighbors = math.isqrt(len(y_test))
    print(neighbors)

    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("Accuracy with k = ", neighbors, " is: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # predicted probabilities of the positive class
    y_scores = knn.predict_proba(X_test)[:, 1]
    # calculate AUC
    print('AUC:', roc_auc_score(y_test, y_scores))

    # generate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(matrix)

# *************Logistic regression***************
def LR():
    print("******Logistic regression********")

    # since we removed the outliers in previous steps 
    # we can use min-max insted of z-score to scale the data

    # create the model
    LRmodel = LogisticRegression(max_iter=1000)
    # define min max scaler
    scaler = MinMaxScaler()
    # transform data
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.fit_transform(X_test)

    # fit the model on the training data
    LRmodel.fit(x_train_scaled, y_train)
    # make predictions on the test data
    y_pred = LRmodel.predict(x_test_scaled)

    # evaluate the model performance
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # predicted probabilities of the positive class
    y_scores = LRmodel.predict_proba(x_test_scaled)[:, 1]
    # calculate AUC
    print('AUC:', roc_auc_score(y_test, y_scores))

    # generate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(matrix)


# #*************Decision Tree***************
def DT():
    print("******Decision Tree********")

    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # evaluate the model performance
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # generate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(matrix)

    plt.figure(figsize=(30, 15))

    a = tree.plot_tree(clf, feature_names=X.columns.values,
                       class_names=['yes', 'no'], rounded=True,
                       filled=True, fontsize=10)
    plt.show()


# #*************SVM***************
def SVM():
    print("******SVM********")
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # evaluate the model performance
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # generate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(matrix)


# NB()
KNN()
# LR()
# DT()
# SVM()


