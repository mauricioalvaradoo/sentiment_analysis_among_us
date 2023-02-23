import pandas as pd
import pickle
import gzip

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    classification_report, confusion_matrix, roc_curve
)

import utils

import warnings
warnings.simplefilter("ignore")




# Importación
df_train = pd.read_pickle('Data/df_train_raw.pkl')
df_test  = pd.read_pickle('Data/df_test_raw.pkl')

# Definición de elementos
X_train = df_train['comments']
y_train = df_train['label']
X_test  = df_test['comments']
y_test  = df_test['label']

# Procesando los textos
X_train = utils.preprocess(X_train)
X_test  = utils.preprocess(X_test)

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




## Modelos ####################################################################
"""
Se comparará algoritmos de ML y estructuras de DL.
Se buscará estimar el mejor modelo. Si se requiere, se usará GridSearchCV.
La comparativa se hará mediante logloss, f1-score, ajuste, curva ROC.

Machine Learning:
* (1) Regresión Logística
* (2) XGBoost

Deep Learning:
* (3) Red Neuronal 1: Intermedia
    + Embedding: 10 000 dimensiones en vocabulario
    + Convolucional: 32 núcleos
    + DropOut: 0.6
    + Pooling: Max
    + Dropout: 0.2
    + Densa: 64 neuronas
    + Densa: 1 neurona

* (4) Red Neuronal 2: Compleja
    + Embedding: 20 000 dimensiones en vocabulario
    + Convolucional: 64 núcleos
    + DropOut: 0.8
    + Pooling: Max
    + Dropout: 0.2
    + Densa: 64 neuronas
    + Densa: 32 neuronas
    + Densa: 1 neurona
    
Todas con 50 epochs.
Acaba la estimación si se alcanza 0.01 de error o 99.5% de ajuste

"""




# (1) Regresión Logística #####################################################
params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Regularización
}

lr = GridSearchCV(LogisticRegression(), param_grid=params, cv=5, verbose=2)\
    .fit(X_train, y_train)
yhat_proba_lr = lr.predict_proba(X_test)[:,1]
yhat_lr       = lr.predict(X_test)

print('Hiperparámetros\t:', lr.best_params_)
print('Ajuste\t:',          lr.best_score_)

# Estadisticos
score_lr = accuracy_score(y_test, yhat_lr)
loss_lr  = log_loss(y_test, yhat_lr)
f1_lr    = f1_score(y_test, yhat_lr)

# Confusion matrix y ROC
conf_lr   = confusion_matrix(y_test, yhat_lr, normalize='true')
report_lr = classification_report(y_test, yhat_lr)
flr, tlr, thresholds = roc_curve(y_test, yhat_proba_lr)




# (2) XGBoost #################################################################
params = {
    'n_estimators':  [100, 200, 500],  # Cantidad de árboles
    'max_depth':     [3, 5, 7],        # Máxima profundida de árboles
    'learning_rate': [0.01, 0.001]     # Learning rate
}

xgb = GridSearchCV(XGBClassifier(), param_grid=params, cv=5, verbose=2)\
    .fit(X_train, y_train)
yhat_proba_xgb = xgb.predict_proba(X_test)[:,1]
yhat_xgb = xgb.predict(X_test)

print('Hiperparámetros\t:', xgb.best_params_)
print('Ajuste\t:',          xgb.best_score_)

# Estadisticos
score_xgb = accuracy_score(y_test, yhat_xgb)
loss_xgb  = log_loss(y_test, yhat_xgb)
f1_xgb    = f1_score(y_test, yhat_xgb)

# Confusion matrix y ROC
conf_xgb = confusion_matrix(y_test, yhat_xgb, normalize='true')
report_xgb = classification_report(y_test, yhat_xgb)
fxgb, txgb, thresholds = roc_curve(y_test, yhat_proba_xgb)




# (3) Red Neuronal 1 ##########################################################
rn1 = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(5_000, 64),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(0.6), 
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

rn1.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.001),
    metrics = ["logcosh", "accuracy"]
)

rn1.fit(
    X_train, y_train, epochs=50,
    validation_data=(X_test, y_test)
)

# Evaluación
# scores = rn1.evaluate(X_test, y_test, verbose=0)




# (4) Red Neuronal 2 ##########################################################
rn2 = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(10_000, 64),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(0.8), 
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

rn2.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.001),
    metrics = ["logcosh", "accuracy"]
)

rn2.fit(
    X_train, y_train, epochs=50,
    validation_data=(X_test, y_test)
)

# Evaluación
# scores = rn2.evaluate(X_test, y_test, verbose=0)




# Comparativa #################################################################

# tf.keras.utils.plot_model(rn1, to_file='rn1.png', show_shapes=True, show_layer_names=True)









# Guardando modelos ###########################################################
with gzip.open('Modelos/lr.pklz', 'wb') as f:
    pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)
with gzip.open('Modelos/xgb.pklz', 'wb') as f:
    pickle.dump(xgb, f, protocol=pickle.HIGHEST_PROTOCOL)

# Las redes neuronales serán guardadas con su propio método 'save' debido a que
# tensorflow maneja su propio estilo de serialización.
rn1.save('Modelos/rn1.h5') 
rn2.save('Modelos/rn2.h5') 

