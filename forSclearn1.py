# -*- coding: utf-8 -*-
"""
Задача класифікації - прогнозування класу аварійності свердловини за її параметрами
Те саме що forSclearn0.py, але для пошуку оптимальних параметрів використовували differential_evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('123_2017_part.csv',sep=';') # прочитати з файлу csv
df=df[['Pump', 'Stress', 'Gas', 'Curv', 'Water', 'H', 'Product', 'Paraffin', 'H19', 'H22', 'H25', 'Month', 'Kv_Cat']]

df=df.dropna()
X=df.drop(['Kv_Cat'], axis=1)
y=df['Kv_Cat']

from sklearn.utils import shuffle
X, y = shuffle(X, y) # випадкове перемішування даних #, random_state=0
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# дані для побудови моделі і екзамену
X_, x_test, y_, y_test = train_test_split(X,y,test_size=0.2)#,random_state=0)

def f(x):
    #n_e, m_d = int(round(x[0])), int(round(x[1]))
    #model=RandomForestClassifier(n_estimators=n_e, max_depth=m_d)
    n_e, l_r, m_d = int(round(x[0])), x[1], int(round(x[2]))
    model=GradientBoostingClassifier(n_estimators=n_e, learning_rate=l_r, max_depth=m_d)
    s=cross_val_score(model, X_, y_, cv=7) # перехресна перевірка
    print s.mean()
    return -s.mean()

from scipy.optimize import differential_evolution
#bounds = [(50, 150), (3, 5)] # границі
bounds = [(50, 150), (0.01, 0.6), (3, 5)] # границі
# закоментуйте наступні 2 рядки, якщо параметри відомі
#res = differential_evolution(f, bounds=bounds)
#print res
##
#model = RandomForestClassifier(n_estimators=121, max_depth=5)
model=GradientBoostingClassifier(n_estimators=120,learning_rate=0.38,max_depth=5)
model.fit(X_,y_)

imp=zip(model.feature_importances_, X.columns.values)
for score, name in sorted(imp,reverse=True):
    print score, name

# звіт по класифікації (виконайте кілька раз і обчисліть середнє)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
Y_test=model.predict(x_test) # екзамен на тестових даних (Мюллер с.279)
print confusion_matrix(y_test, Y_test) # матриця помилок
print classification_report(y_test, Y_test) # повний звіт по класифікації

## крива точності-повноти
# будується для різних порогових значень імовірності
from sklearn.metrics import precision_recall_curve
y_scores=model.predict_proba(x_test)[:,1] # імовірності класу 1 тестових даних
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.plot(precision, recall, 'k-')
for p,r,t in zip(precision[:-1], recall[:-1], thresholds)[::4]: # кожна четверта імовірність
    plt.text(p,r,round(t,2))
plt.xlabel(u"Точність"), plt.ylabel(u"Повнота")
plt.show()
# середня точність класифікатора (площа під кривою точності-повноти)
from sklearn.metrics import average_precision_score
print average_precision_score(y_test, y_scores)