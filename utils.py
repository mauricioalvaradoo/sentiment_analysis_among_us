import nltk
from nltk.corpus import stopwords
import string


stops = stopwords.words('spanish')
punctuations = string.punctuation


def preprocess(y, lemma=False):
    """ Procesamiento de texto
    Parámetros
    --------------
    y [pd.DataFrame]: Texto 
    

    Retorno
    --------------
    y [pd.DataFrame]: Texto procesado

    """

    # Tokenización [list]
    y = y.apply(lambda obs: nltk.word_tokenize(obs))

    # Quitar stopwords y punct [list]
    y = y.apply(lambda obs: [w.lower() for w in obs])
    y = y.apply(lambda obs: [w for w in obs if w.isalpha()])
    y = y.apply(lambda obs: [w for w in obs if w not in stops])
    y = y.apply(lambda obs: [w for w in obs if w not in punctuations])
    
    if lemma == True:
        lematizador = nltk.WordNetLemmatizer()
        y = y.apply(lambda obs: [lematizador.lemmatize(w) for w in obs])
    
    # Unión [lis -> str]
    y = y.apply(lambda obs: ' '.join(obs))
    
    return y
