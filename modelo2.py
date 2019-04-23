

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pickle

stop_words=stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

training_sentences = [
    "Que quieres comer para la cena?",
    "Que quieres comer esta noche?",
    "No se que comer esta noche.",
    "No tendras cangrejos?",
    "Te llevo algo de comer?", 
    "A que hora estaras en casa",
    "Cuanto tiempo estaras?",
    "A que hora nos encontraremos?",
    "Por que estas tomando tanto tiempo?",
    "A que hora estarás aya?",
    "Hola",
    "Hola, como estas?",
    "Buen días",
    "Buenas tardes",
    "Buenas noches" 
]

training_response = [
    "Cualquier comida ",
    "Preferiria cangrejos y salmon",
    "No dispongo de esa comida",
    "No tengo esa comida",
    "Si ,trae algo de comer",
    "En unas horas",
    "En unas horas",
    "Como a las 6",
    "Dificicultades con la hora",
    "A las 4",
    "Hola",
    "Bien y tu",
    "Buenas",
    "Buenas",
    "Buenas" 
]

training_intents = [
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "arrival_time",
    "arrival_time",
    "arrival_time",
    "arrival_time",
    "arrival_time",
    "hello",
    "hello",
    "hello",
    "hello",
    "hello" 
]

import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


# Eliminando comas y otros
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        
    return reviews

reviews_train_clean = preprocess_reviews(training_sentences)

# Eliminando palabras de parada
def second_preprocess (reviews_train_clean):
    clean=[]
    for example_sent in reviews_train_clean:
 
        word_tokens = example_sent.split() 
        
        filtered_sentence = [] 
          
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w) 
        
        clean.append(filtered_sentence)
        
    return clean

reviews_train_clean2 = second_preprocess(reviews_train_clean)

# Lematizacion
def third_preprocess(reviews_train_clean):
    clean=[]
    for example_sent in reviews_train_clean:
        
        filtered_sentence = [] 
          
        for w in example_sent:       
                filtered_sentence.append(stemmer.stem(w))
        
        clean.append(filtered_sentence)
        
    return clean
    

reviews_train_clean3 = third_preprocess(reviews_train_clean2)

'''Si se incluye metodo de bolsa de frases'''
all_text=reviews_train_clean3+third_preprocess(second_preprocess(preprocess_reviews(training_response)))


# Creandon un lexicon
def create_lexicon(reviews_train_clean):
    lexicon = []
    for example_sent in reviews_train_clean:     
        lexicon += list(example_sent)
    # Eliminamos duplicados
    lexicon = list(dict.fromkeys(lexicon))
    return lexicon


#lexicon =create_lexicon(reviews_train_clean3)

lexicon =create_lexicon(all_text)

def sample_handling(reviews_train_clean,lexicon):
    featureset = []
    for example_sent in reviews_train_clean:
        features = np.zeros(len(lexicon))
        for word in example_sent:
            if word in lexicon:
               index_value = lexicon.index(word)
               features[index_value] += 1
        featureset.append(features)
    
    return featureset
        
base_x=sample_handling(reviews_train_clean3,lexicon)


def response_handling(training_intents):
    categories = list(dict.fromkeys(training_intents))
    featureset = []
    for intent in training_intents:
        features = np.zeros(len(categories))
        if intent in categories:
            index_value = categories.index(intent)
            features[index_value] += 1
        featureset.append(features)
    return featureset

def enconde_response(training_intents):
    featureset=[]
    for intent in training_intents:
        if intent[0]== 1:
            res=0
        elif intent[1]== 1:
            res=1
        else:    
            res=2
        featureset.append(res)
    return featureset
            
pre_base_y = response_handling(training_intents)

base_y=enconde_response(pre_base_y)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    base_x, base_y, train_size = 0.8)


for c in [0.01, 0.05, 0.25, 0.5, 1]:    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

lr.coef_

def response(user_input,lexicon,model):
    user_input=sample_handling(third_preprocess(second_preprocess(preprocess_reviews([user_input]))),lexicon)
    if model.predict(user_input)[0]==0:
        res="Pasta con salmon!"
    elif model.predict(user_input)[0]==1:
        res="A las 6 pm."
    else:
        res="Saludos!"
    return res

print(response('Comida Comidaaa',lexicon,lr))
print(response('A que hora llegaras?',lexicon,lr))
print(response('Ques es el tiempo?',lexicon,lr))
print(response('TIEMPOOO',lexicon,lr))
print(response('hora de ir al cine ',lexicon,lr))
print(response('hola hola ',lexicon,lr))
print(response('Buen día ',lexicon,lr))


pickle.dump(lr, open('lr.pkl','wb'))
pickle.dump(lexicon, open('lexicon.pkl','wb'))


''' Segundo Método'''
# volver a correr 


from sklearn.cluster import KMeans
import pandas as pd

resp=sample_handling(third_preprocess(second_preprocess(preprocess_reviews(training_response))),lexicon)

kmeans = KMeans(n_clusters=3, random_state=0).fit(resp)
kmeans.labels_


diccionario=pd.DataFrame({'indice':kmeans.labels_,'respuesta':training_response})

X_train, X_val, y_train, y_val = train_test_split(
    base_x,kmeans.labels_, train_size = 0.8)



for c in [0.01, 0.05, 0.25, 0.5, 1]:    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

def response(user_input,lexicon,model):
    user_input=sample_handling(third_preprocess(second_preprocess(preprocess_reviews([user_input]))),lexicon)
    res=np.random.choice(list(diccionario[diccionario['indice']==model.predict(user_input)[0]]['respuesta']))
    return res


print(response('Comida Comidaaa',lexicon,lr))
print(response('A que hora llegaras?',lexicon,lr))
print(response('Ques es el tiempo?',lexicon,lr))
print(response('TIEMPOOO',lexicon,lr))
print(response('hora de ir al cine ',lexicon,lr))
print(response('hola hola ',lexicon,lr))
print(response('Buen día ',lexicon,lr))
print(response('que es la existencia ',lexicon,lr))
