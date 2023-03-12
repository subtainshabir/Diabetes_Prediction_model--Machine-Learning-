from django.shortcuts import render
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv('C:\\Users\\DELL 5470 i5\\Downloads\\diabetes.csv')
    X = df.drop("Outcome", axis=1)
    Y = df["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    val1 = request.GET['n1']
    val2 = request.GET['n2']
    val3 = request.GET['n3']
    val4 = request.GET['n4']
    val5 = request.GET['n5']
    val6 = request.GET['n6']
    val7 = request.GET['n7']
    val8 = request.GET['n8']

    if val1 == '' or val2 == '' or val3 == '' or val4 == '' or val5 == '' or val6 == '' or val7 == '' or val8 == '':
        result1 = 'Error! You have fill all entreis.'

    else:
        pred = model.predict([[float(val1), float(val2), float(val3), float(
            val4), float(val5), float(val6), float(val7), float(val8)]])

        if pred == 1:
            result1 = "Positive"
        else:
            result1 = "Negative"

    return render(request, 'predict.html', {'result2': result1})
