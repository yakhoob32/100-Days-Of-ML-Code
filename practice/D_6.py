import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = LogisticRegression()
classifier.fit(x_train, Y_train)

y_pred = classifier.predict(x_test)


cm = confusion_matrix(Y_test, y_pred)