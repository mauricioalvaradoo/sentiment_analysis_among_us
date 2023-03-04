import pandas as pd
import matplotlib.pyplot as plt

import spacy
# !pip install spacy
# !python -m spacy download es_core_news_sm

import utils



df_to_classify = pd.read_pickle('Data/df_to_classify_raw.pkl')
X_to_classify  = df_to_classify['comments']


# Procesamiento y vectorización
X_to_classify  = utils.preprocess(X_to_classify, lemma=False)
df_to_classify['procesado'] = X_to_classify




## Estadísticas ###############################################################
# Total de comentarios
df_to_classify.shape[0]

# Cantidad de palabras
len(df_to_classify['procesado'].str.cat(sep=' ').split())


# Frecuencia de palabras por score     
fig = df_to_classify['score'].value_counts().sort_index()

ax = fig.plot(kind='bar')
ax.set_xticklabels(fig.index, rotation=0)
ax.set_ylabel('cantidad de comentarios')

plt.savefig('Figuras/comentarios_por_score.png')
plt.savefig('Figuras/comentarios_por_score.pdf')
plt.show()


# Frecuencia de palabras por score     
fig = df_to_classify.groupby('score')['procesado'].apply(lambda x: len(' '.join(x).split()))
fig = fig/1_000

ax = fig.plot(kind='bar')
ax.set_xticklabels(fig.index, rotation=0)
ax.set_ylabel('miles de palabras')

plt.savefig('Figuras/palabras_por_score.png')
plt.savefig('Figuras/palabras_por_score.pdf')
plt.show()


# Part-of-Speech
nlp = spacy.load('es_core_news_sm')

df_to_classify['pos'] = df_to_classify['procesado'].apply(
    lambda x: [(w.text, w.pos_) for w in nlp(x)]
)

'''
ADJ: Adjetivo
ADP: Adposición
ADV: Adverbio
AUX: Verbo auxiliar
CCONJ: Conjunción coordinada
DET: Determinante
INTJ: Interjección
NOUN: Sustantivo
NUM: Numeral
PART: Partícula
PRON: Pronombre
PROPN: Nombre propio
PUNCT: Puntuación
SCONJ: Conjunción subordinada
SYM: Símbolo
VERB: Verbo
X: Otro
'''

# Cantidad de verbos
## Por por fila
df_to_classify['pos'].apply(
    lambda x: len([w for w in x if w[1] == 'VERB'])
)

## Total
df_to_classify['pos'].apply(
    lambda x: len([w for w in x if w[1] == 'VERB'])
).sum()


# Cantidad de adjetivos
## Por fila
df_to_classify['pos'].apply(
    lambda x: len([w for w in x if w[1] == 'ADJ'])
)

## Total
df_to_classify['pos'].apply(
    lambda x: len([w for w in x if w[1] == 'ADJ'])
).sum()


# TOP 5 personas u organizaciones
pers_orga = []

for doc in nlp.pipe(df_to_classify['procesado']):
    for p in doc.ents:
        pers_orga.append((p.text, p.label_))
        
df_pers_orga = pd.DataFrame(pers_orga, columns=['persona/organización', 'tipo'])

# Filtrado
df_pers_orga = df_pers_orga[df_pers_orga['tipo'].isin(['PER', 'ORG'])]

contado = df_pers_orga['persona/organización'].value_counts()



