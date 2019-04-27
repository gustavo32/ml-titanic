#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# g = sns.FacetGrid(titanic_data, row="Sex", col="Pclass",
#                   hue="Survived", margin_titles=True, aspect=1.2)
# g.map(sns.distplot, "Age", kde=False, bins=bins, hist_kws=dict(alpha=0.7))

def mislabeled_class(estimator, train, X, y):
    train_data = train.copy()
    famSize = train_data["SibSp"] + train_data["Parch"] + 1
    train_data["FamSize"] = np.where(famSize == 1, 0, np.where(famSize==2, 1, np.where(famSize<=4, 2, np.where(famSize>4, 3, -1))))
    train_data.drop(columns=["SibSp", "Parch"], inplace=True)
    cols = ["Pclass", "Sex", "Embarked", "FamSize"]
    y_predict = estimator.predict(X)
    error = abs(y-y_predict)
    
    train_data["Correct"] = 1
    indices = np.argwhere(error == 1).reshape(-1,)
    train_data.iloc[indices, train_data.columns.get_loc("Correct")] = 0
    
    n_rows, n_cols = 2, 2
    
    for i in range(n_rows):
        for j in range(n_cols):
            z=i*n_cols + j
            g = sns.catplot(x=cols[z], y=cols[z], estimator=lambda x: len(x) / len(train_data), 
                      hue="Correct", data=train_data, kind="bar", aspect=0.8, orient="v", legend_out=False)
            g.set_axis_labels("", "Survival Rate")
            g.fig.set_size_inches(8, 5)
            g.set_titles(cols[z])

    train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
    
    bins = np.arange(0, 80, 5)
    z = sns.FacetGrid(train_data, row="Sex", col="Pclass",
                      hue="Correct", margin_titles=True, aspect=1.2)
    z.map(sns.distplot, "Age", kde=False, bins=bins, hist_kws=dict(alpha=0.8))
    z.add_legend()
    plt.show()
    
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    #1º argumento, qual cor em decimal se inicia. 
    #2º argumento qual cor se acabará.
    #3º (as_cmap) Suavizar a cor de acordo com o número.
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, #altura da barra ao lado
        ax=ax,
        annot=True, #colocar os números em cada frame
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Correlação de Pearson através das características', y=1.05, size=15)

# In[2]:

class AddNullAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.astype(np.str)
        return np.where(X == "nan", 1, 0)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''Input a DataFrame and returns given columns on NumPy array'''
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns].values

class CabinFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.astype(np.str)
        return np.where(X == "nan", 0, 1)

class AddTreatment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(categories='auto', sparse=False)
    def create_treatment_column(self, single_data):
        """This function create a treatment's list:
        1 - Miss.
        2 - Mr.
        3 - Mrs.
        4 - Master
        5 - Others"""
#         if re.findall(r"(.+(M|m)iss.+)|(.+(M|m)ile.+)|(.+(M|m)s.+)", single_data):
#             return "Miss"
        if re.findall(r"(.+(M|m)rs.+)|(.+(M|m)me.+)", single_data):
            return "Mrs"
        elif re.findall(r".+(M|m)r.+", single_data):
            return "Mr"
        elif re.findall(r".+(M|m)aster.+", single_data):
            return "Master"
        return "Others"
    def create_symbol_name_column(self, single_data):
        if re.findall(r"(.+\(.+)", single_data):
            return 1
        else:
            return 0
    def fit(self, X, y=None):
        func = np.vectorize(self.create_treatment_column)
        treatment = func(X)
        self.encoder.fit(treatment)
        return self
    def transform(self, X):
        func = np.vectorize(self.create_treatment_column)
        treatment = func(X)
        treat = self.encoder.transform(treatment)
        symbol_func = np.vectorize(self.create_symbol_name_column)
        symbol = symbol_func(X)
        return np.c_[treat, symbol]
        
# In[3]:

class AddAttributes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        famSize = X[:, 0] + X[:, 1] + 1
        group = np.where(famSize == 1, 0, np.where(famSize==2, 1, np.where(famSize<=4, 2, np.where(famSize>4, 3, -1))))
        group = group.reshape(-1, 1)
#         age_risk = np.where((X[:, 2] > 18.) & (X[:, 2] < 36.), 1, 0)
        return group

class ImputerByRegression(BaseEstimator, TransformerMixin):
    def __init__(self, feature, columns, estimator):
        self.feature = feature
        self.columns = columns
        self.estimator = estimator
    def fit(self, X, y=None):
        missing_values = X[X[self.feature].isnull()]
        input_values = X[X[self.feature].notnull()]
        
        print(input_values)

        features = input_values[self.columns].values
        labels = input_values[self.feature].values
        
        est = self.estimator.fit(features, labels)
        return est

    def transform(self, X):
        X[self.feature].where(X[self.feature].notnull(), self.estimator.predict(X[self.columns]), inplace=True)
        return X