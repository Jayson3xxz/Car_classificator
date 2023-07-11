import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
DataSet = pd.read_csv(r"C:\Users\jayson3xxx\Desktop\Training_DataSeTs\Automobile.csv" , low_memory = False )
DataSet.dropna(inplace = True)
print(DataSet.info())
DataSet.drop_duplicates()
DataSet['origin'] = DataSet['origin'].factorize()[0]
print(DataSet['origin'].unique())
DataSet['name'] = DataSet['name'].factorize()[0]
X_dt = DataSet[['horsepower']].copy()
Y_dt = DataSet[['origin']].copy()
sns.heatmap(X_dt.corr(method = 'spearman') , xticklabels = X_dt.corr(method = 'spearman').columns ,
            yticklabels = X_dt.corr(method = 'spearman').columns,cmap = 'coolwarm' , center = 0 , annot = True)
plt.show()
X_np = X_dt.to_numpy()
Y_np = Y_dt.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
x_train , x_test , y_train , y_test = train_test_split(X_st,Y_np)
Model = KNeighborsClassifier().fit(x_train , y_train)
print("Количество правильных ответов на обучающей выборке  : ", Model.score(x_train,y_train))
print("Количество правильных ответов на тестовой выборке  : ", Model.score(x_test,y_test))
print("Подборка лучших параметрову ........",'\n')
params = {'n_neighbors': [i for i in range(1, 51)],
          'weights': ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid = GridSearchCV(estimator= Model, param_grid=params)
grid.fit(x_train,y_train)
print('Наилучшая доля при подборке наиболее подходящих параметров ', grid.best_score_)
print('Наиболее подходящее количество соседей', grid.best_estimator_.n_neighbors)
print('Наилучшая весовая функция', grid.best_estimator_.weights)
print('Наилучший алгоритм поиска ближайших соседей', grid.best_estimator_.algorithm)
#Тест с наилучшими параметрами
New_Model = KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors,
                                 weights=grid.best_estimator_.weights,
                                 algorithm=grid.best_estimator_.algorithm).fit(x_train, y_train)
print("Доля правильных ответов на обучающей выборке с лучшими параметрами :" , New_Model.score(x_train,y_train))
print("Доля правильных ответов на тестовой выборке выборке с лучшими параметрами :" , New_Model.score(x_test,y_test))