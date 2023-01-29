import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# calculating the outliers for each column
for column in df.columns:
    if df[column].dtype == "float64":
        Q1 = float(df[column].quantile(0.25))
        Q3 = float(df[column].quantile(0.75))

        IQR = float(Q3 - Q1)
        n1 = Q1 - (1.5 * IQR)
        n2 = Q3 + (1.5 * IQR)
        outliers = df[(df[column] < n1) | (df[column] > n2)]
        df = df[~df.index.isin(outliers.index)]

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

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(corr_matrix, ax=ax)

plt.show()