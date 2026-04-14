import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""#####Import Dataset#####"""
data = pd.read_csv('Iris.csv')
data.head(5)

"""#####Check Null Values#####"""
data.isnull().sum()
data.describe()

labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for label in labels:
    print('Distribution of', label)
    sns.boxplot(data[label])
    plt.show()

sns.heatmap(data.corr())

"""#####Check if any feature is skewed or not#####"""
labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for label in labels:
    print('Distribution of', label)
    sns.distplot(data[label])
    plt.show()

labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for label in labels:
    print('skewness of ', label)
    print(data[label].skew())