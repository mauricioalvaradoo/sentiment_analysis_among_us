import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

import utils


# Importación
df_train       = pd.read_pickle('Data/df_train_raw.pkl')
df_test        = pd.read_pickle('Data/df_test_raw.pkl')

# Definición de elementos
X_train = df_train['comments']
y_train = df_train['label']
X_test  = df_test['comments']
y_test  = df_test['label']

# Procesando los textos
X_train        = utils.processing(X_train)
X_test         = utils.processing(X_test)

# Vectorización -> Asociado a X_train
vectorizer  = TfidfVectorizer(
    max_features = 5_000,
    use_idf = True,
    ngram_range = (1, 2)
)

X_train = vectorizer.fit_transform(X_train).toarray()
X_test  = vectorizer.transform(X_test).toarray()

# Guardado el vectorizador
with open('Data/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)



## Modelos ##################################################################
"""
Se comparará algoritmos de ML y estructuras de DL.
La comparativa se hará mediante MAPE y ajuste.
Los mejores modelos por algoritmo serán tomados por GridSearchCV.

Machine Learning
* (1) Regresión Logística
* (2) XGBoost
* (3) SVM

Deep Learning:
* (4)
* (5)
* (6)


"""


# (1) Regresión Logística ###################################################
lr = LogisticRegression(random_state=19).fit(X_train, y_train)
yhat_proba_lr = lr.predict_proba(X_test)[:,1]
yhat_lr = lr.predict(X_test)

# Estadisticos
score_lr = accuracy_score(y_test, yhat_lr)
ps_lr = precision_score(y_test, yhat_lr)
r_lr = recall_score(y_test, yhat_lr)
f1_lr = f1_score(y_test, yhat_lr)

# Confusion matrix y ROC
conf_lr = confusion_matrix(y_test, yhat_lr, normalize="true")
report_lr = classification_report(y_test, yhat_lr)
flr, tlr, thresholds = roc_curve(y_test, yhat_proba_lr)



# (2) XGBoost ################################################################
xgb = XGBClassifier(use_label_encoder=False, eval_metric = "mlogloss", random_state = 19)\
    .fit(X_train, y_train)
yhat_proba_xgb = xgb.predict_proba(X_test)[:,1]
yhat_xgb = xgb.predict(X_test)

# Estadisticos
score_xgb = accuracy_score(y_test, yhat_xgb)
ps_xgb = precision_score(y_test, yhat_xgb)
r_xgb = recall_score(y_test, yhat_xgb)
f1_xgb = f1_score(y_test, yhat_xgb)

# Confusion matrix y ROC
conf_xgb = confusion_matrix(y_test, yhat_xgb, normalize="true")
report_xgb = classification_report(y_test, yhat_xgb)
fxgb, txgb, thresholds = roc_curve(y_test, yhat_proba_xgb)


# (3) SVM ####################################################################
svm_grid_search = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1],
    'kernel': ['rbf'],
    'probability': [True],
    'random_state': [19]
}

grid = GridSearchCV(
    svm_grid_search, param_grid, cv = 5, scoring = "accuracy",
    verbose = 10, n_jobs = -1
)

svm = grid.fit(X_train, y_train)
yhat_proba_svm = svm.predict_proba(X_test)[:,1]
yhat_svm = svm.predict(X_test)

# Estadisticos
score_svm = accuracy_score(y_test, yhat_svm)
ps_svm = precision_score(y_test, yhat_svm)
r_svm = recall_score(y_test, yhat_svm)
f1_svm = f1_score(y_test, yhat_svm)

# Confusion matrix y ROC
conf_svm = confusion_matrix(y_test, yhat_svm, normalize="true")
report_svm = classification_report(y_test, yhat_svm)
fsvm, tsvm, thresholds = roc_curve(y_test, yhat_proba_svm)




# (4) Red Neuronal 1 #########################################################
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10_000, 64),
    tf.keras.layers.Conv1D(100, 3, activation='relu'),
    tf.keras.layers.Dropout(0.8), 
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=2)

