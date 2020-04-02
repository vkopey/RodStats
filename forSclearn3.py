# -*- coding: utf-8 -*-
"""
Задача регресії - - прогнозування частоти аварій свердловини за її параметрами
Те саме що forSclearn2.py, але для пошуку оптимальних параметрів використовували differential_evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('123_2017_part.csv',sep=';') # прочитати з файлу csv
df=df[['Pump', 'Stress', 'Gas', 'Curv', 'Water', 'H', 'Product', 'Paraffin', 'Kv', 'H19', 'H22', 'H25', 'Month']]

df=df.dropna()
#print df
X=df.drop(['Kv'], axis=1)
y=df['Kv']

from sklearn.utils import shuffle
X, y = shuffle(X, y) # випадкове перемішування даних

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# дані для побудови моделі і екзамену
X_, x_test, y_, y_test = train_test_split(X,y,test_size=0.2)#,random_state=0)

def f(x):
    #n_e, m_d = int(round(x[0])), int(round(x[1]))
    #model=RandomForestRegressor(n_estimators=n_e, max_depth=m_d)
    n_e, l_r, m_d = int(round(x[0])), x[1], int(round(x[2]))
    model=GradientBoostingRegressor(n_estimators=n_e, learning_rate=l_r, max_depth=m_d)
    s=cross_val_score(model, X_, y_, cv=7) # перехресна перевірка
    print s.mean()
    return -s.mean()

from scipy.optimize import differential_evolution
#bounds = [(50, 150), (3, 5)] # границі
bounds = [(50, 150), (0.01, 0.6), (3, 5)] # границі
# закоментуйте наступні 2 рядки, якщо параметри моделі уже відомі
#res = differential_evolution(f, bounds=bounds) 
#print res
##
# найкраща модель
#model = RandomForestRegressor(n_estimators=133, max_depth=5)
model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.15, max_depth=4)
model.fit(X_, y_) # найкраща модель для даних для навчання
Y_test=model.predict(x_test) # екзамен на тестових даних (Мюллер с.279)
from sklearn.metrics import r2_score
print r2_score(y_test, Y_test) # або model.score(x_test, y_test)

plt.scatter(y_test, Y_test) # реальні і прогнозовані тестові дані
plt.xticks(np.arange(1,16))
#plt.yticks(np.arange(0,17))
#plt.axis('equal')
plt.grid()
plt.show()

imp=zip(model.feature_importances_, X.columns.values)
for score, name in sorted(imp,reverse=True):
    print score, name

## виконати ці рядки з консолі, а потім виконати програму N раз для перехресної перевірки
#yt=dict.fromkeys(range(1,16))
#for i in yt: yt[i]=[]
##
for y,Y in zip(y_test, Y_test):
    yt[y].append(Y)

# зберігає у файл csv. Потім відкрити в Excel, замінити '.' на ',', зберегти і передати в Statistica 10 для аналізу
df = pd.DataFrame() # об'єкт DataFrame
for i in yt: # кожний стопчик
    df[i] = pd.Series(yt[i]) # об'єкт Series
df.to_csv('yt_Yt.csv',sep=';',index=False,header=False,mode='w+') # увага! режим додавання до файлу 

# print np.mean(yt[1]), np.std(yt[1])    
# plt.figure()
# plt.hist(yt[1]) # гістограма
# plt.show()
