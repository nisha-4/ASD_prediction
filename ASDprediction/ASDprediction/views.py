from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    # DATA_COLLECTION AND ANALYSIS
    asd_dataset = pd.read_csv(r'\asd.csv')
    # removing nan value to zero
    asd_dataset.replace(np.nan, 0)
    # Separating data and label
    x = asd_dataset.drop(columns='syndromic', axis=1)
    y = asd_dataset['syndromic']
    # Data Standardization
    scalar = StandardScaler()  # loading the standard scaler function into variable
    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    # training the model
    classifier = svm.SVC(kernel='linear')
    asd_dataset['chromosome'] = X = 50;
    asd_dataset['chromosome'] = Y = 51;
    asd_dataset['chromosome'] = asd_dataset.chromosome.astype(str).astype('Int64')
    x_train = x_train.replace((np.inf, -np.inf, np.NaN), 0).reset_index(drop=True)
    x_test = x_test.replace((np.inf, -np.inf, np.NaN), 0).reset_index(drop=True)

    # Random Forest
    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
    classifier.fit(x_train, y_train)

    #######################Taking Value From Html Form #######################################
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])

    pred = classifier.predict([[val1 , val2, val3, val4 , val5]])

    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    # # Prediction the test set result
    # y_pred = classifier.predict(x_test)
    #
    # accuracyrf = confusion_matrix(y_test, y_pred)
    #
    # print(accuracyrf)
    # accrf = ((accuracyrf[0][0] + accuracyrf[1][1]) / (
    #             accuracyrf[0][0] + accuracyrf[1][1] + accuracyrf[1][0] + accuracyrf[0][1])) * 100
    # print(accrf)

    return render(request, 'predict.html' , {"result2":result1})
