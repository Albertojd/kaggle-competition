# Kaggle-competition

Competición en Kaggle sobre modelos de predicción

En este proyecto tenemos dos csv, "train" y "predict", lo primero que hacemos es limpiarlos, y exportar a unos nuevos csv (se puede ver de manera más espcifca en el archivo "CleanData.ipynb").

Los dos metodos usados para predecir los datos fueron con las librerias H2O y RandomForestRegressor().

A continuación con más detalle:

## H2O:
H2O es un producto creado por la compañía H2O.ai con el objetivo de combinar los principales algoritmos de machine learning y aprendizaje estadístico con el Big Data. Gracias a su forma de comprimir y almacenar los datos, H2O es capaz de trabajar con millones de registros en un único ordenador (emplea todos sus cores) o en un cluster de muchos ordenadores. 

El código usado para para desarrolar esta librería se encuentra en Model_H2O

## RandomForestRegressor():

Para elegir este metodo he testeado varios metodos: 
```python
models = {
    "forest" : RandomForestRegressor(),
    "tree" : DecisionTreeRegressor(),
    "neighbors_reg": KNeighborsRegressor(),
    "gradient": GradientBoostingRegressor()
    "KNeighbors":KNeighborsClassifier(),
    "decision" : SVC()
    }

for name, model in models.items():
    print(f"Training {name}")
    model.fit(X_train, y_train)
print("Finish")

# Output: 
-------forest-------
RMSE 540.365
-------tree-------
RMSE 751.255

from sklearn.metrics import mean_squared_error
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"-------{name}-------")
    print ("RMSE", round(np.sqrt(mean_squared_error(y_test,y_pred)),3))

# Output: 
forest accuracy: 0.9813076134179681
tree accuracy: 0.9639390717248901
```
El siguiente paso es hiperparametrizar el mejor metodo con la función "GridSearchCV", que ayuda a recorrer hiperparámetros predefinidos y selecciona los mejores parámetros para un modelo en concreto:

```python
parameters = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200]}

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)

grid = GridSearchCV(rfr, params, n_jobs=-1, verbose=1)
grid.fit(X_train,y_train)
```

## Ejecutar el modelo con los parametros indicados:

Por úlitmo usamos el modelo con elegido con los parametros que nos indica GridSearchCV:

```python
	id	price
0	0	890.976834
1	1	6576.103596
2	2	712.330697
3	3	1891.560655
4	4	946.098256
```

