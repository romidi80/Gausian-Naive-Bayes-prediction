import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("Iris.csv")
df = pd.DataFrame(data, index=[i for i in range(len(data) - 1)]).reset_index(drop=True)
X = df.drop(columns = ["Class"])
Y = df["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
GNB = GaussianNB()
GNB.fit(X_train, Y_train)
Y_predict = GNB.predict(X_test)
print("Confusion matrix :")
print(confusion_matrix(Y_test, Y_predict))
print("Accuracy :", accuracy_score(Y_test, Y_predict))