import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        # df[column].fillna('Missing', inplace=True)

# calculating the outliers for each column
outliers = pd.DataFrame()
for column in df.columns:
    if df[column].dtype == "float64":
        Q1 = float(df[column].quantile(0.25))
        Q3 = float(df[column].quantile(0.75))

        IQR = float(Q3 - Q1)
        n1 = Q1 - (1.5 * IQR)
        n2 = Q3 + (1.5 * IQR)
        column_outliers = df.loc[(df[column] < n1) | (df[column] > n2)]
        column_outliers["column"] = column
        outliers = outliers.append(column_outliers)

# outliers.to_excel('outliers.xls')

plt.boxplot(df['MaxTemp'])
plt.title('Box plot of {}'.format('MaxTemp'))
tick_positions = np.arange(start=min(df['MaxTemp']), stop=max(df['MaxTemp']), step=2)
plt.yticks(tick_positions)
plt.show()



