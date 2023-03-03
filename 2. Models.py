import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
)

from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    classification_report, confusion_matrix, roc_curve
)

import utils

import warnings
warnings.simplefilter('ignore')




# Definición de elementos
df_train = pd.read_pickle('Data/df_train_raw.pkl')
df_test  = pd.read_pickle('Data/df_test_raw.pkl')

X_train = df_train['comments']
y_train = np.array(df_train['label'])
X_test  = df_test['comments']
y_test  = np.array(df_test['label'])


# Procesando los textos #######################################################
X_train_ml = utils.preprocess(X_train, lemma=True )
X_test_ml  = utils.preprocess(X_test,  lemma=True )
X_train_dl = utils.preprocess(X_train, lemma=False)
X_test_dl  = utils.preprocess(X_test,  lemma=False)


# Vectorización para modelos ML ###############################################
# La vectorización SÍ considera Lemmatization
vectorizer  = TfidfVectorizer(
    max_features = 15_000,
    use_idf = True
)

X_train_ml = vectorizer.fit_transform(X_train_ml).toarray()
X_test_ml  = vectorizer.transform(X_test_ml).toarray()

# Guardado el vectorizador
with open('Data/vectorizer-ml.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)



# Vectorización para modelos DL ###############################################
# La vectorización NO considera Lemmatization. Esto por las capas Embedding
tokenizer = Tokenizer(num_words=15_000)
tokenizer.fit_on_texts(X_train_dl)

X_train_dl = tokenizer.texts_to_sequences(X_train_dl)
X_test_dl  = tokenizer.texts_to_sequences(X_test_dl)
X_train_dl = pad_sequences(X_train_dl, padding='post', maxlen=100)
X_test_dl  = pad_sequences(X_test_dl,  padding='post', maxlen=100)

# Guardado el tokenizador
with open('Data/vectorizer-dl.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)




## Modelos ####################################################################
"""
Se comparará algoritmos de ML y estructuras de DL.
Se buscará estimar el mejor modelo. Si se requiere, se usará GridSearchCV.
La comparativa se hará mediante ajuste, logloss, f1-score, curva ROC.

Machine Learning:
* (1) Regresión Logística
* (2) XGBoost

Deep Learning:
* (3) Convolucional Red Neuronal 1: Básica
    + Embedding: 15 000 dimensiones en vocabulario
    + Convolucional: 32 núcleos
    + Pooling: Global Max
    + Dropout: 0.5
    + Densa: 128 neuronas
    + Densa: 32 neuronas
    + Densa: 1 neurona

* (4) Convolucional Red Neuronal 2: Intermedia
    + Embedding: 15 000 dimensiones en vocabulario
    + Convolucional: 64 núcleos
    + Pooling: Global Max
    + Dropout: 0.3
    + Densa: 128 neuronas
    + Densa: 64 neuronas
    + Densa: 1 neurona

Todas con 20 epochs.

"""




# (1) Regresión Logística #####################################################
params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Regularización
}

lr = GridSearchCV(LogisticRegression(), param_grid=params, cv=5, verbose=2)\
    .fit(X_train_ml, y_train)
yhat_proba_lr = lr.predict_proba(X_test_ml)[:,1]
yhat_lr       = lr.predict(X_test_ml)

print('Hiperparámetros\t:', lr.best_params_)
print('Ajuste\t:',          lr.best_score_)

# Estadisticos
score_lr = accuracy_score(y_test, yhat_lr)
loss_lr  = log_loss(y_test, yhat_lr)
error_lr = np.mean(y_test != yhat_lr)
f1_lr    = f1_score(y_test, yhat_lr)

# Confusion matrix y ROC
conf_lr   = confusion_matrix(y_test, yhat_lr, normalize='true')
report_lr = classification_report(y_test, yhat_lr)
flr, tlr, thresholds = roc_curve(y_test, yhat_proba_lr)




# (2) XGBoost #################################################################
params = {
    'n_estimators':  [200, 400], # Cantidad de árboles
    'max_depth':     [5, 7, 9],  # Máxima profundida de árboles
    'learning_rate': [0.01]      # Learning rate
}

xgb = GridSearchCV(XGBClassifier(), param_grid=params, cv=5, verbose=2)\
    .fit(X_train_ml, y_train)
yhat_proba_xgb = xgb.predict_proba(X_test_ml)[:,1]
yhat_xgb       = xgb.predict(X_test_ml)

print('Hiperparámetros\t:', xgb.best_params_)
print('Ajuste\t:',          xgb.best_score_)
# results_cv     = xgb.cv_results_
# params_cv      = results_cv['params']
# mean_scores_cv = results_cv['mean_test_score']
# std_scores_cv  = results_cv['std_test_score']
# for i in range(len(params_cv)):
#     print(params_cv[i], " -- mean score:", mean_scores_cv[i], " -- std score:", std_scores_cv[i])


# Estadisticos
score_xgb = accuracy_score(y_test, yhat_xgb)
loss_xgb  = log_loss(y_test, yhat_xgb)
error_xgb = np.mean(y_test != yhat_xgb)
f1_xgb    = f1_score(y_test, yhat_xgb)

# Confusion matrix y ROC
conf_xgb = confusion_matrix(y_test, yhat_xgb, normalize='true')
report_xgb = classification_report(y_test, yhat_xgb)
fxgb, txgb, thresholds = roc_curve(y_test, yhat_proba_xgb)




# (3) Red Neuronal 1 ##########################################################
rn1 = tf.keras.Sequential(
    [
        Embedding(input_dim=15_000, output_dim=128, input_length=100, name='embedding_1'),
        
        Conv1D(filters=32, kernel_size=3, activation='relu', name='conv_1'),      
        GlobalMaxPooling1D(),
        
        Dropout(0.5),
        Dense(units=128, activation='relu', name='dense_1'),
        Dense(units=32, activation='relu', name='dense_2'),
        Dense(units=1, activation='sigmoid', name='dense_3')
    ]
)

rn1.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.01),
    metrics = ['logcosh', 'accuracy']
)

history_rn1 = rn1.fit(
    X_train_dl, y_train, epochs=30,
    validation_data=(X_test_dl, y_test)
)

yhat_proba_rn1 = rn1.predict(X_test_dl).flatten()
yhat_rn1       = (yhat_proba_rn1>0.5).astype('int32')

# Estadisticos
score_rn1 = accuracy_score(y_test, yhat_rn1)
loss_rn1  = log_loss(y_test, yhat_rn1)
error_rn1 = np.mean(y_test != yhat_rn1)
f1_rn1    = f1_score(y_test, yhat_rn1)

# Confusion matrix y ROC
conf_rn1   = confusion_matrix(y_test, yhat_rn1, normalize='true')
report_rn1 = classification_report(y_test, yhat_rn1)
frn1, trn1, thresholds = roc_curve(y_test, yhat_proba_rn1)




# (4) Red Neuronal 2 ##########################################################
rn2 = tf.keras.Sequential(
    [
        Embedding(input_dim=15_000, output_dim=128, input_length=100, name='embedding_1'),
        
        Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_1'),      
        GlobalMaxPooling1D(),
        
        Dropout(0.3),
        Dense(units=128, activation='relu', name='dense_1'),
        Dense(units=64, activation='relu', name='dense_2'),
        Dense(units=1, activation='sigmoid', name='dense_3')
    ]
)

rn2.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.01),
    metrics = ['logcosh', 'accuracy']
)

history_rn2 = rn2.fit(
    X_train_dl, y_train, epochs=30,
    validation_data=(X_test_dl, y_test)
)

yhat_proba_rn2 = rn2.predict(X_test_dl).flatten()
yhat_rn2       = (yhat_proba_rn2>0.5).astype('int32')

# Estadisticos
score_rn2 = accuracy_score(y_test, yhat_rn2)
loss_rn2  = log_loss(y_test, yhat_rn2)
error_rn2 = np.mean(y_test != yhat_rn2)
f1_rn2    = f1_score(y_test, yhat_rn2)

# Confusion matrix y ROC
conf_rn2   = confusion_matrix(y_test, yhat_rn2, normalize='true')
report_rn2 = classification_report(y_test, yhat_rn2)
frn2, trn2, thresholds = roc_curve(y_test, yhat_proba_rn2)




# Guardando modelos ###########################################################
with gzip.open('Modelos/lr.pklz', 'wb') as f:
    pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)
with gzip.open('Modelos/xgb.pklz', 'wb') as f:
    pickle.dump(xgb, f, protocol=pickle.HIGHEST_PROTOCOL)

# Las redes neuronales serán guardadas con su propio método 'save' debido a que
# tensorflow maneja su propio estilo de serialización.
rn1.save('Modelos/rn1.h5') 
rn2.save('Modelos/rn2.h5') 



# Recuperando modelos #########################################################
# with gzip.open('Modelos/lr.pklz', 'rb') as f:
#     lr = pickle.load(f, protocol=pickle.HIGHEST_PROTOCOL)
# with gzip.open('Modelos/xgb.pklz', 'rb') as f:
#     xgb = pickle.load(f, protocol=pickle.HIGHEST_PROTOCOL)

# rn1.load('Modelos/rn1.h5') 
# rn2.load('Modelos/rn2.h5') 




# Comparativa #################################################################
## 1. Ajuste
dict_ajust = {
    'Regresión Logística': score_lr,
    'XGBoost': score_xgb,
    'CNN Básica': score_rn1,
    'CNN Intermedia': score_rn2,
}

df_score = pd.DataFrame(dict_ajust.items(), columns=['Modelo', 'Score']).set_index('Modelo')
df_score['Score'] = np.round(df_score['Score'], 3)
df_score = df_score.sort_values('Score', ascending=False)

print(df_score)


## 2. LogLoss
dict_loss = {
    'Regresión Logística': loss_lr,
    'XGBoost': loss_xgb,
    'CNN Básica': loss_rn1,
    'CNN Intermedia': loss_rn2,
}

df_loss = pd.DataFrame(dict_loss.items(), columns=['Modelo', 'LogLoss']).set_index('Modelo')
df_loss['LogLoss'] = np.round(df_loss['LogLoss'], 3)
df_loss = df_loss.sort_values('LogLoss', ascending=True)

print(df_loss)


## 3. F1-Score
dict_f1 = {
    'Regresión Logística': f1_lr,
    'XGBoost': f1_xgb,
    'CNN Básica': f1_rn1,
    'CNN Intermedia': f1_rn2,
}

df_f1 = pd.DataFrame(dict_f1.items(), columns=['Modelo', 'F1-Score']).set_index('Modelo')
df_f1['F1-Score'] = np.round(df_f1['F1-Score'], 3)
df_f1 = df_f1.sort_values('F1-Score', ascending=False)

print(df_f1)


## 4. Curva ROC
dict_roc = {
    'Regresión logistica': [flr, tlr],
    'XGBoost': [fxgb, txgb],
    'CNN Básica': [frn1, trn1],
    'CNN Intermedia': [frn2, trn2]
}
keys   = list(dict_roc.keys())
values = list(dict_roc.values())


# Figura
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label='(ROC = 0.5)')

j = 0
for i in values:
    plt.plot(i[0], i[1], label=keys[j])
    j += 1

plt.xlabel('Ratio falso positivo')
plt.ylabel('Ratio verdadero positivo')
plt.title('Curvas ROC')
plt.legend(fontsize=9)

plt.savefig('Figuras/roc.pdf', bbox_inches='tight', transparent=True)
plt.savefig('Figuras/roc.png', bbox_inches='tight', transparent=True)
plt.show()




# Guardando estadísticas ######################################################
with open('Data/dict_ajust.pkl', 'wb') as f:
    pickle.dump(dict_ajust, f)
with open('Data/dict_loss.pkl', 'wb') as f:
    pickle.dump(dict_loss, f)
with open('Data/dict_f1.pkl', 'wb') as f:
    pickle.dump(dict_f1, f)
with open('Data/dict_roc.pkl', 'wb') as f:
    pickle.dump(dict_roc, f)

