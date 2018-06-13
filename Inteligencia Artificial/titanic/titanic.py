# LIBRERÍAS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import scale
# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# LEER ARCHIVOS
data_train = pd.read_csv('C:/Users/agus_/Downloads/train.csv')
data_test = pd.read_csv('C:/Users/agus_/Downloads/test.csv')

# Información del dataset completo
print(data_train.info())
print("-"*40)
print(data_test.info())
print("-"*67)
print(data_train.describe())
print("\n")

# Features originales del dataset
print(data_train.columns.values)
print("-"*35)
print(data_test.columns.values)
print("\n")

# ETAPAS DE ANÁLISIS DE DATOS - INGENIERÍA DE FEATURES
# Se analizarán aquellos features que consideramos necesarios para incluirlos en nuestro modelo. Para ello, se seguirá
# una serie de pasos para luego decidir qué features son relevantes y cuales no.

# 1) Correlación de features
# En esta etapa, analizaremos los features que creemos que tienen correlación con Survived. Solo haremos esto con aquellas
# características que no tengan valores vacíos. En caso de tener una alta correlación, se incluirán en el modelo.

print(data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="Pclass", y="Survived", data=data_train, kind="bar", size=6 , palette="muted")
grid.despine(left=True)
grid = grid.set_ylabels("survival probability")
plt.show()

print(data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="Sex", y="Survived", data=data_train,kind="bar", size=6 , palette="muted")
grid.despine(left=True)
grid = grid.set_ylabels("survival probability")
plt.show()

print(data_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="SibSp", y="Survived", data=data_train, kind="bar", size=6 , palette="muted")
grid.despine(left=True)
grid = grid.set_ylabels("survival probability")
plt.show()

print(data_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="Parch", y="Survived", data=data_train, kind="bar", size=6 , palette="muted")
grid.despine(left=True)
grid = grid.set_ylabels("survival probability")
plt.show()

print(data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="Embarked", y="Survived", data=data_train, size=6, kind="bar", palette="muted")
grid.despine(left=True)
grid = grid.set_ylabels("survival probability")
plt.show()

# sns.set(style="darkgrid")
grid = sns.FacetGrid(data_train, col='Survived')
grid = grid.map(sns.distplot, 'Age', hist=True, hist_kws=dict(edgecolor="w"), color='blue')
plt.show()

# 2) Corrección de features
# En esta etapa, se eliminarán aquellos features que se consideran totalmente irrelevantes para incluirlos en el modelo.
# ¿Cómo nos damos cuenta de ello? Simple, se observan aquellos features que son independientes y no aportan información
# para saber si la persona sobrevivió o no. En este caso, son PassengerId, Ticket y Cabin.

data_train = data_train.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
data_test = data_test.drop(['Ticket', 'Cabin'], axis=1)

print(data_train.columns.values)
print(data_train.shape)
print("\n")
print(data_test.columns.values)
print(data_test.shape)
print("\n")

# 3) Creación de features
# En esta etapa, se analizarán aquellos features que por si solos hacen que el modelo sea más complejo, pero agrupando
# esas características en una nueva, simplifica el modelo y ayuda a entenderlo aún más.
# Se analizará si es conveniente crear una nueva característica a partir de las existentes.

dataset = [data_train, data_test]
for data in dataset:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

for data in dataset:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

print(data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n")
grid = sns.factorplot(x="Title", y="Survived", data=data_train, kind="bar")
grid = grid.set_xticklabels(["Master","Miss", "Mrs","Mr","Rare"])
grid = grid.set_ylabels("survival probability")
plt.show()

transformacion_de_titulos = {"Master": 1, "Miss": 2, "Mrs": 3, "Mr": 4, "Other": 5}
for data in dataset:
    data['Title'] = data['Title'].map(transformacion_de_titulos)
    data['Title'] = data['Title'].fillna(value=0) # fillna() ---> busca todos los valores NaN y los reemplaza por 0

print()

data_train = data_train.drop(['Name'], axis=1)
data_test = data_test.drop(['Name'], axis=1)
dataset = [data_train, data_test]

# Sex dummies
data_train = pd.get_dummies(data=data_train, columns=['Sex'])
data_train = data_train.drop(['Sex_male'], axis=1)
data_test = pd.get_dummies(data=data_test, columns=['Sex'])
data_test = data_test.drop(['Sex_male'], axis=1)
dataset = [data_train, data_test]

print(data_train.columns.values)
print(data_train.head())
print(data_test.columns.values)
print(data_test.head())
print("\n")
# print(data_train.info())
print("-"*60)

# Completando Age
sumaEdadMaster = 0.0
sumaEdadMr = 0.0
sumaEdadMiss = 0.0
sumaEdadMrs = 0.0
sumaEdadOther = 0.0
master = 0
miss = 0
mrs = 0
mr = 0
other = 0
for row in data_train.itertuples(index=True):
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMaster = sumaEdadMaster + getattr(row, 'Age')
        master += 1
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMiss = sumaEdadMiss + getattr(row, 'Age')
        miss += 1
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMrs = sumaEdadMrs + getattr(row, 'Age')
        mrs += 1
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMr = sumaEdadMr + getattr(row, 'Age')
        mr += 1
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadOther = sumaEdadOther + getattr(row, 'Age')
        other += 1
    # print(getattr(row, 'Title'), getattr(row, 'Age'))

for row in data_test.itertuples(index=True):
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMaster = sumaEdadMaster + getattr(row, 'Age')
        master += 1
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMiss = sumaEdadMiss + getattr(row, 'Age')
        miss += 1
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMrs = sumaEdadMrs + getattr(row, 'Age')
        mrs += 1
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadMr = sumaEdadMr + getattr(row, 'Age')
        mr += 1
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == False:
        sumaEdadOther = sumaEdadOther + getattr(row, 'Age')
        other += 1
    # print(row[['Title', 'Age']])

print("SUMA:", sumaEdadMaster, "CANT:", master)
media_master = sumaEdadMaster/master
print("MEDIA Master:", media_master)
print("SUMA:", sumaEdadMiss, "CANT:", miss)
media_miss = sumaEdadMiss/miss
print("MEDIA Miss:", media_miss)
print("SUMA", sumaEdadMrs, "CANT:", mrs)
media_mrs = sumaEdadMrs/mrs
print("MEDIA Mrs:", media_mrs)
print("SUMA:", sumaEdadMr, "CANT:", mr)
media_mr = sumaEdadMr/mr
print("MEDIA Mr:", media_mr)
print("SUMA:", sumaEdadOther, "CANT:", other)
media_other = sumaEdadOther/other
print("MEDIA Other:", media_other)
print("TOTAL:", master+miss+mrs+mr+other)
print("\n")

print(data_train.info())
print("\n")

for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index ,'Age'] = media_master
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index, 'Age'] = media_miss
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index, 'Age'] = media_mrs
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index, 'Age'] = media_mr
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index, 'Age'] = media_other
    # print(getattr(row, 'Title'), getattr(row, 'Age'))

# Convertir todos los valores del feature Age en números enteros. De float a int.
data_train['Age'] = data_train['Age'].astype(np.int64)

for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_master
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_miss
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_mrs
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_mr
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_other

# Convertir todos los valores del feature Age en números enteros. De float a int.
data_test['Age'] = data_test['Age'].astype(np.int64)

print(data_train.info())
print(data_train.head())
print("\n")
print(data_test.info())
print(data_test.head())
print("\n")

dataset = [data_train, data_test]
print(data_train.shape)
print(data_test.shape)
print("\n")

data_train['AgeRange'] = pd.cut(data_train['Age'], 8)
print(data_train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True))
data_train = data_train.drop(['AgeRange'], axis=1)
print("\n")

for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Age') <= 10:
        data_train.at[index, 'Age'] = 0
    if getattr(row, 'Age') > 10 and getattr(row, 'Age') <= 20:
        data_train.at[index, 'Age'] = 1
    if getattr(row, 'Age') > 20 and getattr(row, 'Age') <= 30:
        data_train.at[index, 'Age'] = 2
    if getattr(row, 'Age') > 30 and getattr(row, 'Age') <= 40:
        data_train.at[index, 'Age'] = 3
    if getattr(row, 'Age') > 40 and getattr(row, 'Age') <= 50:
        data_train.at[index, 'Age'] = 4
    if getattr(row, 'Age') > 50 and getattr(row, 'Age') <= 60:
        data_train.at[index, 'Age'] = 5
    if getattr(row, 'Age') > 60 and getattr(row, 'Age') <= 70:
        data_train.at[index, 'Age'] = 6
    if getattr(row, 'Age') > 70:
        data_train.at[index, 'Age'] = 7

print(data_train.head())
print("\n")

for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Age') <= 10:
        data_test.at[index, 'Age'] = 0
    if getattr(row, 'Age') > 10 and getattr(row, 'Age') <= 20:
        data_test.at[index, 'Age'] = 1
    if getattr(row, 'Age') > 20 and getattr(row, 'Age') <= 30:
        data_test.at[index, 'Age'] = 2
    if getattr(row, 'Age') > 30 and getattr(row, 'Age') <= 40:
        data_test.at[index, 'Age'] = 3
    if getattr(row, 'Age') > 40 and getattr(row, 'Age') <= 50:
        data_test.at[index, 'Age'] = 4
    if getattr(row, 'Age') > 50 and getattr(row, 'Age') <= 60:
        data_test.at[index, 'Age'] = 5
    if getattr(row, 'Age') > 60 and getattr(row, 'Age') <= 70:
        data_test.at[index, 'Age'] = 6
    if getattr(row, 'Age') > 70:
        data_test.at[index, 'Age'] = 7

print(data_test.head())
print("-"*60)

# Completando Embarcaciones
embarked_C = 0
embarked_S = 0
embarked_Q = 0
for row in data_train.itertuples(index=True):
    if getattr(row, 'Embarked') == 'C' and pd.isnull(getattr(row, 'Embarked')) == False:
        embarked_C += 1
    if getattr(row, 'Embarked') == 'S' and pd.isna(getattr(row, 'Embarked')) == False:
        embarked_S += 1
    if getattr(row, 'Embarked') == 'Q' and pd.isna(getattr(row, 'Embarked')) == False:
        embarked_Q += 1

for row in data_test.itertuples(index=True):
    if getattr(row, 'Embarked') == 'C' and pd.isnull(getattr(row, 'Embarked')) == False:
        embarked_C += 1
    if getattr(row, 'Embarked') == 'S' and pd.isna(getattr(row, 'Embarked')) == False:
        embarked_S += 1
    if getattr(row, 'Embarked') == 'Q' and pd.isna(getattr(row, 'Embarked')) == False:
        embarked_Q += 1

print("Cantidad de pasajeros que embarcaron en C:", embarked_C)
print("Cantidad de pasajeros que embarcaron en S:", embarked_S)
print("Cantidad de pasajeros que embarcaron en Q:", embarked_Q)
print("\n")

for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if pd.isna(getattr(row, 'Embarked')) ==  True:
        data_train.at[index, 'Embarked'] = 'S'

dataset = [data_train, data_test]
for data in dataset:
    data['Embarked'] = data['Embarked'].replace(['S'], 0)
    data['Embarked'] = data['Embarked'].replace(['Q'], 1)
    data['Embarked'] = data['Embarked'].replace(['C'], 2)

print(data_train.head())
print("\n")
print(data_test.head())
print("-"*60)

# Completando Fare
data_test['Fare'].fillna(data_test['Fare'].dropna().mean(), inplace=True)
print(data_test.info())
print("\n")

data_train['Fare'] = data_train['Fare'].astype(np.int64)
data_test['Fare'] = data_test['Fare'].astype(np.int64)
print(data_test.info())
print("\n")

data_train['FareRange'] = pd.qcut(data_train['Fare'], 5, duplicates='drop')
print(data_train[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True))
grid = sns.factorplot(x="FareRange", y="Survived", data=data_train, kind="bar")
grid = grid.set_xticklabels(["0", "1", "2", "3", "4"])
grid = grid.set_ylabels("survival probability")
plt.show()
data_train = data_train.drop(['FareRange'], axis=1)

for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Fare') <= 7:
        data_train.at[index, 'Fare'] = 0
    if getattr(row, 'Fare') > 7 and getattr(row, 'Fare') <= 10:
        data_train.at[index, 'Fare'] = 1
    if getattr(row, 'Fare') > 10 and getattr(row, 'Fare') <= 21:
        data_train.at[index, 'Fare'] = 2
    if getattr(row, 'Fare') > 21 and getattr(row, 'Fare') <= 39:
        data_train.at[index, 'Fare'] = 3
    if getattr(row, 'Fare') > 39:
        data_train.at[index, 'Fare'] = 4

for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Fare') <= 7:
        data_test.at[index, 'Fare'] = 0
    if getattr(row, 'Fare') > 7 and getattr(row, 'Fare') <= 10:
        data_test.at[index, 'Fare'] = 1
    if getattr(row, 'Fare') > 10 and getattr(row, 'Fare') <= 21:
        data_test.at[index, 'Fare'] = 2
    if getattr(row, 'Fare') > 21 and getattr(row, 'Fare') <= 39:
        data_test.at[index, 'Fare'] = 3
    if getattr(row, 'Fare') > 39:
        data_test.at[index, 'Fare'] = 4

print(data_train.head())
print("n")
print(data_test.head())
print("\n")
print("-"*60)

# Creación de un nuevo feature: IsAlone
data_train['FamilySize'] = None
for row in data_train.itertuples(index=True):
    data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1

print(data_train.head())
print(data_train.info())
print("\n")

data_test['FamilySize'] = None
for row in data_test.itertuples(index=True):
    data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch'] + 1

print(data_test.head())
print(data_test.info())
print("\n")

data_train['IsAlone'] = None
for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female, FamilySize, IsAlone = row
    if getattr(row, 'FamilySize') != 1:
        # Tiene familia
        data_train.at[index, 'IsAlone'] = 0
    else:
        # Está solo
        data_train.at[index, 'IsAlone'] = 1

print(data_train.head())
print("\n")
print(data_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
print("\n")

data_test['IsAlone'] = None
for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female, FamilySize, IsAlone = row
    if getattr(row, 'FamilySize') != 1:
        # Tiene familia
        data_test.at[index, 'IsAlone'] = 0
    else:
        # Está solo
        data_test.at[index, 'IsAlone'] = 1

print(data_test.head())
print("\n")

data_train = data_train.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
print(data_train.head())
print("\n")
data_test = data_test.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
print(data_test.head())
print("\n")

# APLICACIÓN DE LOS DIFERENTES ALGORITMOS DE CLASIFICACIÓN

# data_train = data_train.astype(float)
X = data_train.drop(['Survived'], axis=1)
X = X.astype(np.float64)
y = data_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)

# y_train = data_train["Survived"]
# X_train = data_train.drop("Survived", axis=1)
# X_test = data_test.drop("PassengerId", axis=1).copy()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(y.shape)
print("\n")
print(X.head())
print("\n")
print(y.head())
print("\n")

# Support Vector Machines sin normalización de datos
svc = SVC()
svc.fit(X_train, y_train)
y_hat = svc.predict(X_test)
print("Tasa de aciertos para SVM sin normalización de datos" , round(svc.score(X_train, y_train) * 100, 2))

# Randon Forest sin normalización de datos
random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
y_hat = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Tasa de aciertos para Random Forest sin normalización de datos" , acc_random_forest)
print("\n")

# Support Vector Machines con normalización de datos
media = X_train.values.mean(axis=0)
desviacion = np.std(X_train.values, 0)
i = 0
j = 0

# Declaración del conjunto de entrenamiento normalizado
X_train_norm = np.zeros(np.shape(X_train.values))
# Declaración del conjunto de test normalizado
X_test_norm = np.zeros(np.shape(X_test.values))

# Normalización del conjunto de entrenamiento
X_train_norm = stats.zscore(X_train.values)
# Normalización del conjunto de test
for i in range(X_test.values.shape[1]): #shape 0 te da las filas
    for j in range(X_test.values.shape[0]):
        X_test_norm[j,i] = (X_test.values[j,i] - media[i])/(desviacion[i])

svc = SVC(probability=True)
svc.fit(X_train_norm, y_train)
y_hat = svc.predict(X_test_norm)
print("Tasa de aciertos para SVM con normalización de datos" , round(svc.score(X_train_norm, y_train) * 100, 2))

# Cálculo de la matriz de probabilidades para SVM con datos normalizados
y_hat_proba = svc.predict_proba(X_train_norm)
print("El área bajo la curva es:", metrics.roc_auc_score(y_true=y_test, y_score=y_hat))
print("\n")
# Reporte de clasificación (precision, recall, f1-score y support)
print(metrics.classification_report(y_test, y_hat, target_names=['Not Survived', 'Survived']))
skplt.metrics.plot_roc(y_train, y_hat_proba) # gráfico
plt.show()
print("Confusion Matrix y_hat")
print("----------------------")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_hat, labels=[0,1]))
skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_hat, labels=[0,1])
plt.show()
#result.Survived = svc.predict(X_test)

# Randon Forest con normalización de datos
random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train_norm, y_train)
y_hat = random_forest.predict(X_test_norm)
acc_random_forest = round(random_forest.score(X_train_norm, y_train) * 100, 2)
print("Tasa de aciertos para Random Forest con normalización de datos" , acc_random_forest)

# Cálculo de la matriz de probabilidades para Random Forest con datos normalizados
y_hat_proba = random_forest.predict_proba(X_train_norm)
print("El área bajo la curva es:", metrics.roc_auc_score(y_true=y_test, y_score=y_hat))
print("\n")
# Reporte de clasificación (precision, recall, f1-score y support)
print(metrics.classification_report(y_test, y_hat, target_names=['Not Survived', 'Survived']))
skplt.metrics.plot_roc(y_train, y_hat_proba) # gráfico
plt.show()
print("Confusion Matrix y_hat")
print("----------------------")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_hat, labels=[0,1]))
skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_hat, labels=[0,1])
plt.show()

ids = data_test['PassengerId']
predicciones = random_forest.predict(data_test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predicciones })
output.to_csv('prediccion-titanic.csv', index = False)
print(output.head())
# ------------------------------------------------------------------------------------------------------

# PCA
pca = PCA(n_components=7)

# Antes de transformar los datos, los mismos deben estar normalizados
# scale() estandariza los datos con respecto a la media 0 y a la desv. estándar 1 (z-score)

X_scaled = scale(X)
pca.fit(X_scaled)
# X_transformed son los datos X transformados linealmente con respecto a los componentes principales
X_transformed = pca.transform(X_scaled)
# Veamos los vectores de componentes de PCA...
print('Componentes de PCA (ordenados desc. desde el 1° hasta el 7° vector): \n\n', pca.components_)
# Notar que, por la restricción de la ortogonalidad, la máxima
# cantidad de componentes principales es la de los features de X
print('Varianza explicada por cada componente: \n\n', pca.explained_variance_)

# Visualizamos ahora cuánto es explicada la varianza
# por cada uno de los componentes principales
y_pos = np.arange(7)
# pca.explained_variance_ratio_ es quien nos devuelve el gráfico de la varianza
plt.bar(y_pos, np.round(100 * pca.explained_variance_ratio_, decimals=1), align='center', alpha=0.5)
plt.xticks(y_pos, [1, 2, 3, 4, 5, 6, 7])
plt.xlabel('N° de Componente Principal')
plt.ylabel('Varianza')
plt.title('Varianza explicada por cada componente')
plt.show()

plt.plot(X_transformed[y==0, 0], np.zeros(len(X_transformed[y==0, 0])), 'o', label=0, color='red')
plt.plot(X_transformed[y==1, 0], np.zeros(len(X_transformed[y==1, 0])), 'o', label=1, color='green')


plt.xlabel('Valor del Primer Componente Principal')
plt.legend(loc='best', numpoints=1)
plt.show()

# Incluimos el segundo componente principal, vemos que no cambia sustancialmente...
plot = plt.scatter(X_transformed[y==0, 0], X_transformed[y==0, 1], label=y[0], color='red')
plot = plt.scatter(X_transformed[y==1, 0], X_transformed[y==1, 1], label=y[1], color='green')

plt.xlabel('Valor del Primer Componente Principal')
plt.ylabel('Valor del Segundo Componente Principal')
plt.legend(loc='best', numpoints=1)
plt.show()

# código agregado para graficar los vectores
V = np.array([[pca.components_[0,0],pca.components_[1,0]],
              [pca.components_[0,1],pca.components_[1,1]],
              [pca.components_[0,2],pca.components_[1,2]],
              [pca.components_[0,3],pca.components_[1,3]],
              [pca.components_[0,4],pca.components_[1,4]],
              [pca.components_[0,5],pca.components_[1,5]],
              [pca.components_[0,6],pca.components_[1,6]]])
origin = [0], [0] # origin point

plot = plt.scatter(X_transformed[y==0, 0], X_transformed[y==0, 1], label='Not Survived', color='red')
plot = plt.scatter(X_transformed[y==1, 0], X_transformed[y==1, 1], label='Survived', color='green')

# notar que los vectores están agrandados a modo de mejor visualización
plt.quiver(*origin, V[:,0], V[:,1], color=['black','purple','blue','yellow','red','orange','pink'], scale=3)

plt.text(pca.components_[0,0],pca.components_[1,0], data_train.columns[1], fontsize=12, weight=1000)
plt.text(pca.components_[0,1],pca.components_[1,1], data_train.columns[2], fontsize=12, weight=1000)
plt.text(pca.components_[0,2],pca.components_[1,2], data_train.columns[3], fontsize=12, weight=1000)
plt.text(pca.components_[0,3],pca.components_[1,3], data_train.columns[4], fontsize=12, weight=1000)
plt.text(pca.components_[0,4],pca.components_[1,4], data_train.columns[5], fontsize=12, weight=1000)
plt.text(pca.components_[0,5],pca.components_[1,5], data_train.columns[6], fontsize=12, weight=1000)
plt.text(pca.components_[0,6],pca.components_[1,6], data_train.columns[7], fontsize=12, weight=1000)
plt.xlabel('Valor del Primer Componente Principal')
plt.ylabel('Valor del Segundo Componente Principal')
plt.legend(loc='best', numpoints=1)
plt.show()

# PCA con Random Forest
X_train_PCA, X_test, y_train, y_test = train_test_split(X_transformed, y, random_state=10, test_size=0.3)

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train_PCA, y_train)
y_hat = random_forest.predict(X_test_norm)
acc_random_forest = round(random_forest.score(X_train_PCA, y_train) * 100, 2)
print("Tasa de aciertos para Random Forest con PCA" , acc_random_forest)

 #No se justifica realizar la transformacion  de los features con PCA ya que la varianza de cada componente
# que generamos no varia en gran medida entre ellos y para llegar a un 80 % de la varianza de los datos
# se necesitan como minimo 5 componentes de 7 que se generan. Por otro lado al aplicar PCA en uno de los clasificadores
# notamos que disminuye la tasa de aciertos. Esta es otra causa q nos determina que no es significativo aplicar PCA