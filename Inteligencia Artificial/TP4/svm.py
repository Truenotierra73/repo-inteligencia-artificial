# LIBRERÍAS
import pandas as pd
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
grid = grid.map(sns.distplot, 'Age', hist=True, hist_kws=dict(edgecolor="w"), color='green')
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

print(data_train.head(n=10))

data_train = data_train.drop(['Name'], axis=1)
data_test = data_test.drop(['Name'], axis=1)
dataset = [data_train, data_test]

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
# for row in dataset._iter_():
#     if row.get_values('Title') == 1 and pd.isna(row.get_values('Age')) == False:
#         sumaEdadMaster = sumaEdadMaster + getattr(row, 'Age')
#         master += 1
#     if row['Title'] == 1 and pd.isna(row['Age']) == False:
#         sumaEdadMiss = sumaEdadMiss + getattr(row, 'Age')
#         miss += 1
#     if row['Title'] == 1 and pd.isna(row['Age']) == False:
#         sumaEdadMrs = sumaEdadMrs + getattr(row, 'Age')
#         mrs += 1
#     if row['Title'] == 1 and pd.isna(row['Age']) == False:
#         sumaEdadMr = sumaEdadMr + getattr(row, 'Age')
#         mr += 1
#     if row['Title'] == 1 and pd.isna(row['Age']) == False:
#         sumaEdadOther = sumaEdadOther + getattr(row, 'Age')
#         other += 1
#     # print(row[['Title', 'Age']])


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

for row in data_train.itertuples(index=True):
    index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == True:
        data_train.at[index ,'Age'] = media_master
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == True:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = media_miss
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == True:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = media_mrs
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == True:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = media_mr
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == True:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = media_other
    # print(getattr(row, 'Title'), getattr(row, 'Age'))

# Convertir todos los valores del feature Age en números enteros. De float a int.
data_train['Age'] = data_train['Age'].astype(np.int64)

for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Title') == 1 and pd.isna(getattr(row, 'Age')) == True:
        data_test.at[index, 'Age'] = media_master
    if getattr(row, 'Title') == 2 and pd.isna(getattr(row, 'Age')) == True:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = media_miss
    if getattr(row, 'Title') == 3 and pd.isna(getattr(row, 'Age')) == True:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = media_mrs
    if getattr(row, 'Title') == 4 and pd.isna(getattr(row, 'Age')) == True:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = media_mr
    if getattr(row, 'Title') == 5 and pd.isna(getattr(row, 'Age')) == True:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = media_other

# Convertir todos los valores del feature Age en números enteros. De float a int.
data_test['Age'] = data_test['Age'].astype(np.int64)

print(data_train.info())
print(data_train.head(n=891))
print(data_test.info())
print(data_test.head(n=418))

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
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 1
    if getattr(row, 'Age') > 20 and getattr(row, 'Age') <= 30:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 2
    if getattr(row, 'Age') > 30 and getattr(row, 'Age') <= 40:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 3
    if getattr(row, 'Age') > 40 and getattr(row, 'Age') <= 50:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 4
    if getattr(row, 'Age') > 50 and getattr(row, 'Age') <= 60:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 5
    if getattr(row, 'Age') > 60 and getattr(row, 'Age') <= 70:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 6
    if getattr(row, 'Age') > 70:
        # index, Survived, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_train.at[index, 'Age'] = 7

print(data_train.head())
print("\n")

for row in data_test.itertuples(index=True):
    index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
    if getattr(row, 'Age') <= 10:
        data_test.at[index, 'Age'] = 0
    if getattr(row, 'Age') > 10 and getattr(row, 'Age') <= 20:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 1
    if getattr(row, 'Age') > 20 and getattr(row, 'Age') <= 30:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 2
    if getattr(row, 'Age') > 30 and getattr(row, 'Age') <= 40:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 3
    if getattr(row, 'Age') > 40 and getattr(row, 'Age') <= 50:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 4
    if getattr(row, 'Age') > 50 and getattr(row, 'Age') <= 60:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 5
    if getattr(row, 'Age') > 60 and getattr(row, 'Age') <= 70:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 6
    if getattr(row, 'Age') > 70:
        # index, PassengerId, Pclass, Age, SibSp, Parch, Fare, Embarked, Title, Sex_female = row
        data_test.at[index, 'Age'] = 7

print(data_test.head())
print("\n")

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


# Completando Fare del conjunto de Test
data_test['Fare'].fillna(data_test['Fare'].dropna().mean(), inplace=True)
print(data_test.info())