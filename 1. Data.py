# !pip install google-play-scraper
# !pip install datasets
import google_play_scraper as gplay
import datasets
import pandas as pd



## Data de Entrenamiento ================================================================
# Se usará el dataset traducido del corpus Stanford Sentiment Treebank (SST-2).
# https://huggingface.co/datasets/mrm8488/sst2-es-mt
train, test = datasets.load_dataset(
    path = 'mrm8488/sst2-es-mt',
    name = 'spanish',
    split = ['train', 'validation']
)

train = pd.DataFrame(train)
test  = pd.DataFrame(test)

# Solo se trabajará con aquellos en español
df_train = train[['sentence_es', 'label']]
df_test  = test [['sentence_es', 'label']]
df_train.rename({'sentence_es': 'comments'}, axis=1, inplace=True)
df_test.rename({'sentence_es': 'comments'}, axis=1, inplace=True)

df_train.to_pickle('Data/df_train_raw.pkl')
df_test.to_pickle('Data/df_test_raw.pkl')



## Data por clasificar ==================================================================
# Como testing se usará los comentarios de Google Play Store sobre el juego Among Us
# Actualizado a la fecha de: .............................
comments = gplay.reviews(
    'com.innersloth.spacemafia', # id
    lang = 'es',                 # language
    sort = gplay.Sort.NEWEST,    # más reciente a más antiguo
    country = 'pe',              # país
    count = 5_000                # observaciones
)[0]


df_to_classify = []
for i in comments:   
    df_to_classify.append(
        {
            'user': i['userName'],
            'comments': i['content'],
            'score': i['score']
        }
    )

df_to_classify = pd.DataFrame(df_to_classify, columns = ['user', 'comments', 'score'])

df_to_classify.to_pickle('Data/df_to_classify_raw.pkl')

