import pandas as pd
import math
import numpy as np

def calculate_mean_cov(data):
    mean = np.zeros((4,1))
    for i in range(len(data)):
        for j in range(4):
            mean[j] += float(data[i][j])
    mean[:] = [x / len(data) for x in mean]
    cov = np.zeros((4, 4))
    for i in range(len(data)):
        for j in range(4):
            for k in range(4):
                cov[j][k] += (float(data[i][j]) - mean[j]) * (float(data[i][k]) - mean[k])
    return mean, cov / ((len(data) - 1) * 4)

def return_argmax(arr):
    max = arr[0][0][0]
    argmax = 0
    for i in range(len(arr)-1):
        if max < arr[i+1][0][0]:
            max = arr[i+1][0][0]
            argmax=i+1
    return argmax

def predict(data,mean,cov):
    probabilities = [0] * 3
    for j in data.keys():
        if j == data.keys()[-1]:
            break
        for k in range(len(mean)):
            probabilities[k] += (-0.5 * ((np.array([[data[j]]*4])) - np.array(mean[k]).transpose()).dot(np.linalg.inv(cov[k])
            .dot(((np.array([[data[j]]*4])) - np.array(mean[k]).transpose()).transpose()))) - ((len(mean[k]) / 2) * np.log(2 * math.pi)) - (0.5 * np.log(np.linalg.det(cov[k])))          
    return return_argmax(probabilities)

data = pd.read_csv("Iris.csv")
df = pd.DataFrame(data, index=[i for i in range(len(data) - 1)]).reset_index(drop=True)

train_data = df.sample(frac = 0.7)
test_data = df.drop(train_data.index)

train_classes = dict()
for i in range(len(train_data)):
    if train_data.iloc(0)[i][4] not in train_classes.keys():
        train_classes[train_data.iloc(0)[i][4]] = list()
    train_classes[train_data.iloc(0)[i][4]].append(train_data.iloc(0)[i])

mean = [None] * len(train_classes)
cov = [None] * len(train_classes)
i = 0
for key in train_classes:
    mean[i], cov[i] = calculate_mean_cov(train_classes[key])
    i += 1

Confusion_Matrix = [None]*len(train_classes)

for i in range(len(train_classes)):
    Confusion_Matrix[i] = dict()
    for j in train_classes.keys():
        Confusion_Matrix[i][j] = 0

predicted = [None]*len(test_data)
for i in range(len(test_data)):
    predicted[i] = predict(test_data.iloc(0)[i], mean,cov)
    Confusion_Matrix[predicted[i]][test_data.iloc(0)[i][4]] += 1

print("confution matrix:") 
for row in range(len(Confusion_Matrix)):
    print(list(Confusion_Matrix[row][col] for col in train_classes.keys()))

true = 0
all = 0
i = 0
for j in train_classes.keys():
    true += Confusion_Matrix[i][j]
    all += sum(Confusion_Matrix[i][k] for k in train_classes.keys())
    i+=1
print("Accuracy:", true / all)
