
import pandas as pd

df = pd.read_csv("/kaggle/input/student-performance-multiple-linear-regression/Student_Performance.csv")

df.head()

df.info()

replace_dict = {'Yes': '1', 'No': '0'}

df['Extracurricular Activities'] = df['Extracurricular Activities'].replace(replace_dict).astype('int')

df.info()

import sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

fig, axes = plt.subplots(3, 2, figsize=(20, 15))

axes = axes.flatten()

for i, column in enumerate(df.columns[:-1]):

    ax = axes[i]
    ax.scatter(df[df.columns[-1]], df[column])
    ax.set_xlabel(column)
    ax.set_ylabel('Performance Index')
    
plt.show()

sns.heatmap(df.corr(), annot=True)

target_col = df.columns[-1]

corr = df.corr()[target_col].abs()

high_corr_cols = corr[corr > 0.2].index

new_df = df[high_corr_cols]

new_df.info()

x = new_df.iloc[:, :-1]

y = new_df.iloc[:, -1]

x.info()

scale = MinMaxScaler().fit(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = scale.transform(x_train)

x_test = scale.transform(x_test)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error

model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("R2 Score : ", r2_score(y_test, predictions))

print("Mean Absolute Error : ", mean_absolute_error(y_test, predictions))
