
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
from flask import Flask, render_template,request#,redirect,url_for
import datetime
import re
import os

stop_words=stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


# Eliminando comas y otros
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews


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


# Lematizacion
def third_preprocess(reviews_train_clean):
    clean=[]
    for example_sent in reviews_train_clean:

        filtered_sentence = []

        for w in example_sent:
                filtered_sentence.append(stemmer.stem(w))

        clean.append(filtered_sentence)

    return clean



# Creandon un lexicon
def create_lexicon(reviews_train_clean):
    lexicon = []
    for example_sent in reviews_train_clean:
        lexicon += list(example_sent)
    # Eliminamos duplicados
    lexicon = list(dict.fromkeys(lexicon))
    return lexicon



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



#def response(user_input,lexicon,model):
#    user_input=sample_handling(third_preprocess(second_preprocess(preprocess_reviews([user_input]))),lexicon)
#    if model.predict(user_input)[0]==0:
#        res="Pasta con salmon!"
#    elif model.predict(user_input)[0]==1:
#        res="A las 6 pm."
#    else:
#        res="Saludos!"
#    return res

def response(user_input,lexicon,model):
    user_input=sample_handling(third_preprocess(second_preprocess(preprocess_reviews([user_input]))),lexicon)
    if model.predict(user_input)[0]==0:
        res="Revisar la tasa de interes en la pagina web"
    elif model.predict(user_input)[0]==1:
        res="Depende de una evaluación crediticia"
    else:
        res="Saludos!"
    return res



''' Importando pickles '''

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir, 'lr.pkl')
with open(pickle_file_path, 'rb') as pickle_file:
    lr = pickle.load(pickle_file)


pickle_file_path = os.path.join(my_dir, 'lexicon.pkl')
with open(pickle_file_path, 'rb') as pickle_file:
    lexicon = pickle.load(pickle_file)


#lr = pickle.load(open('lr.pkl','rb'))
#lexicon = pickle.load(open('lexicon.pkl','rb'))

print(response('Hola',lexicon,lr))

''' Flask '''

app = Flask(__name__)


@app.route('/',methods = ['POST', 'GET'])
def index():
   return render_template('result.html')

@app.route('/process',methods=['POST'])
def process():
    now = datetime.datetime.now()
    time=str(now.hour)+':'+str(now.minute)
    user_input=request.form['user_input']
    bot_response=response(user_input,lexicon,lr)
    bot_response=str(bot_response)
    print("Friend: "+bot_response)
    return render_template('result.html',user_input=user_input,
		bot_response=bot_response,hour=time
		)


if __name__ == '__main__':
   app.run(debug = False)
