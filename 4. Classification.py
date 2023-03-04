import pandas as pd
import matplotlib.pyplot as plt
import pickle
import gzip

import utils



# Importación
with open('Data/vectorizer-ml.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
df_to_classify = pd.read_pickle('Data/df_to_classify_raw.pkl')
X_to_classify  = df_to_classify['comments']


# Procesamiento y vectorización
X_to_classify  = utils.preprocess(X_to_classify, lemma=True)
df_to_classify['procesado'] = X_to_classify

X_to_classify  = vectorizer.transform(X_to_classify).toarray()




# Clasificación ###############################################################
with gzip.open('Modelos/lr.pklz', 'rb') as f:
    lr = pickle.load(f)

yhat_lr = lr.predict(X_to_classify)

df_to_classify['labels'] = yhat_lr




# Conteo ######################################################################
counts = df_to_classify.groupby(['score', 'labels']).size().unstack(fill_value=0)
counts

ax = counts.plot(kind='bar', stacked=False)

ax.set_xticklabels(counts.index, rotation=0)
ax.set_xlabel('score')
ax.set_ylabel('cantidad de comentarios')
plt.legend(['negativo', 'positivo'])

plt.savefig('Figuras/clasificación.pdf')
plt.savefig('Figuras/clasificación.png')
plt.show()



# Guardado ####################################################################
with open('Data/df_clasificated.pkl', 'wb') as f:
    pickle.dump(df_to_classify, f)