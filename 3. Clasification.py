import pandas as pd
import nltk
import pickle

import utils


# Importación
with open('Data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
df_to_classify = pd.read_pickle('Data/df_to_classify_raw.pkl')
X_to_classify  = df_to_classify['comments']


# Procesamiento y vectorización
X_to_classify  = utils.preprocess(X_to_classify)
X_to_classify  = vectorizer.transform(X_to_classify).toarray()



## Estadísticas ###############################################################
# 






## Clasificación ##############################################################