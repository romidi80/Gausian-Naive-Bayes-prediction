import pandas as pd
import math


def calculate_mean_var(data):
    mean = [0] * 4
    var = [0] * 4
    for i in range(len(data)):
        for j in range(4):
            mean[j] += float(data[i][j])
            var[j] += (float(data[i][j]) - mean[j]) ** 2
    var[:] = [math.sqrt(x / (len(data) - 1)) for x in var]
    mean[:] = [x / len(data) for x in mean]
    return mean, var

def predict(data,mean,var):
    probabilities = dict()
    for i in mean.keys():
        for j in range(4):
            if i not in probabilities:
                probabilities[i] = 1
            probabilities[i]*=(math.exp(-0.5*((data[j]-mean[i][j])**2/(var[i][j]**2))))
    return max(probabilities, key = probabilities.get)

data = pd.read_csv("Iris.csv")
df = pd.DataFrame(data, index=[i for i in range(len(data) - 1)]).reset_index(drop=True)

train_data = df.sample(frac = 0.7)
test_data = df.drop(train_data.index)

train_classes = dict()
for i in range(len(train_data)):
    if train_data.iloc(0)[i][4] not in train_classes.keys():
        train_classes[train_data.iloc(0)[i][4]] = list()
    train_classes[train_data.iloc(0)[i][4]].append(train_data.iloc(0)[i])

mean = dict()
var= dict()
for key in train_classes.keys():
    mean[key], var[key] = calculate_mean_var(train_classes[key])

Confusion_Matrix = dict()

for i in train_classes.keys():
    Confusion_Matrix[i] = dict()
    for j in train_classes.keys():
        Confusion_Matrix[i][j] = 0

predicted = [None]*len(test_data)
for i in range(len(test_data)):
    predicted[i] = predict(test_data.iloc(0)[i], mean,var)
    Confusion_Matrix[predicted[i]][test_data.iloc(0)[i][4]] += 1


print("confution matrix:") 
for row in Confusion_Matrix.keys():
    print(list(Confusion_Matrix[row][col] for col in Confusion_Matrix.keys()))


true = 0
all = 0
for i in Confusion_Matrix.keys():
    true += Confusion_Matrix[i][i]
    all += sum(Confusion_Matrix[i][j] for j in Confusion_Matrix.keys())
print("Accuracy:", true / all)