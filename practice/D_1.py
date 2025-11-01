import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('../datasets/Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# کدینگ مقادیر غیر عددی
# --->
LabelEncoder_x = LabelEncoder()
X[:, 0] = LabelEncoder_x.fit_transform(X[:, 0])
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)
# <---


# دسته بندی اعداد نسبت داده شده
# --->
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
# <---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# محاسبه واریانس و میانگین و نزدیک کردن اعداد جدول
# --->
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# <---




regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')


plt.show()