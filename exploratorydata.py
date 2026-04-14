import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""#####Import Dataset#####""" 

# Import dataset
data = pd.read_csv('Titanic Dataset.csv')
data.head(5)

"""##### **Passengers belonging from which gender survived the most**"""
sns.countplot(x='Gender', hue='Survived', data=data)

"""##### **Passengers belonging from which Pclass survived the most and the least**"""
sns.countplot(x='Pclass', hue='Survived', data=data)

"""##### **Highest number of passengers belong to which Age**"""
sns.histplot(data['Age'], kde=False, bins=40)

"""##### **Highest number of passengers belong to which Gender**"""
sns.countplot(x='Gender', data=data)

"""##### **Is SibSp correlated/associated with Survived feature**"""
sns.countplot(x='Survived', hue='SibSp', data=data, palette="mako")

"""##### **Is Parch correlated/associated with Survived feature**"""
sns.countplot(x='Survived', hue='Parch', data=data, palette="mako")

"""##### **Is the feature Fare having normal distribution/spread of data**"""
sns.histplot(data['Fare'])

plt.show()

sns.boxplot(x='Pclass', y='Age', data=data, palette='winter')

"""##### **Check the correlation of all the features with target variable 'Survived'**"""
sns.heatmap(data.corr())