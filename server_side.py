
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pickle
import datetime
import re



from flask import Flask, render_template,request,redirect,url_for
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
   
