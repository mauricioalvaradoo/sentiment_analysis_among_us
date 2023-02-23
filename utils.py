import pandas as pd
import nltk
from nltk.corpus import stopwords
import string


stops = stopwords.words('spanish')
punctuations = string.punctuation


def processing(y):
    """ Procesamiento de texto
    Parámetros
    --------------
    y [pd.Serie]: Texto 

    Retorno
    --------------
    y [pd.Serie]: Texto procesado

    """

    # Tokenización [list]
    y = y.apply(lambda obs: nltk.word_tokenize(obs))

    # Quitar stopwords y punct [list]
    y = y.apply(lambda obs: [w.lower() for w in obs])
    y = y.apply(lambda obs: [w for w in obs if w.isalpha()])
    y = y.apply(lambda obs: [w for w in obs if w not in stops])
    y = y.apply(lambda obs: [w for w in obs if w not in punctuations])

    # Lematización [list]
    lematizador = nltk.WordNetLemmatizer()
    y = y.apply(lambda obs: [lematizador.lemmatize(w, pos='v') for w in obs])

    # Unión [lis -> str]
    y = y.apply(lambda obs: ' '.join(obs))

    return y

