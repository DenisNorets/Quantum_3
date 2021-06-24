import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import sklearn
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
train_data = pd.read_csv('C:/Users/User/Quantum internship/internship_train.csv')
train_data.head()
y_train = train_data.target
y_train.head()
X_train = train_data.iloc[:,:-1].copy()
X_train.head()
X_test = pd.read_csv('C:/Users/User/Quantum internship/internship_hidden_test.csv')
X_test.head()
models = [
    [ linear_model.Lasso(), { 'alpha': [0.5,0.6,0.7,0.8,0.9,1] } ],
    [ linear_model.LinearRegression(), {}]
]
min_score = -1
best_clf = -1
for model in models:
    for cv_try in range (2, 11):
        clf = GridSearchCV(estimator=model[0], param_grid=model[1], scoring='neg_root_mean_squared_error', cv=cv_try)
        clf.fit(X_train, y_train)
        if min_score == -1 or min_score > clf.best_score_*(-1):
            min_score = clf.best_score_*(-1)
            best_clf = clf
print(f'Best classifier: {best_clf.best_estimator_}')
print(f'Scorer function: {best_clf.scorer_}')
print(f'RMSE: {best_clf.best_score_*(-1)}')
result_data = pd.Series(best_clf.predict(X_test))
result_data.to_csv('C:/Users/User/Quantum internship/result_test.csv')

sns.distplot(result_data)

