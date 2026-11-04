import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset & Target Binarization
df = pd.read_csv('student-mat.csv', sep=';')
df['Target'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# 2. Feature Selection & Encoding
features = ['studytime', 'failures', 'absences', 'higher', 'romantic', 'G1', 'G2']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Target']

# 3. Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Model Training
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42).fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)
