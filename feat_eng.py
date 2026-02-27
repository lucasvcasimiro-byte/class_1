import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.head())
print(diabetes.info())

X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=15, shuffle = True, stratify = y)


# Creation
X['High Risk Flag'] = ((X['Glucose']>125) and (X['BMI']>30)).astype('int')
X['Insulin/Glucose'] = X['Insulin']/X['Glucose']







# Scaling
rbs = RobustScaler()

X_train_scl= rbs.fit_transform(X_train)

# Transforming the test and train data:
X_train_scl = rbs.transform(X_train)
X_test_scl = rbs.transform(X_test)


#Plotting scaled data
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(data=X_train, ax=axes[0], color='#0EE071')
axes[0].set_title('Before Scaling (Raw Data)')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=X_train_scl, ax=axes[1], color='#0EE071')
axes[1].set_title('After Robust Scaling (Centered at 0)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Selection