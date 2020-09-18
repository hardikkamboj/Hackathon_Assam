# Importing essential libraries
from flask import Flask, render_template, request
from tensorflow import keras
import json
from nltk.stem.porter import PorterStemmer
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = keras.models.load_model('my_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		my_prediction = predict(message)
		return render_template('result.html', prediction=my_prediction)


def predict(text):

	#importing stop words
	with open("stop_words.txt", "r") as fp:
		stop_words = json.load(fp)

	print(len(stop_words))
	ps=PorterStemmer()

	result = re.sub('[^a-zA-Z]',' ',text)
	result = result.lower()
	result = result.split()
	result = [ps.stem(word) for word in result if not word in stop_words]
	result = ' '.join(result)
	clean_text = result

	vocab_size = 10000
	one_hot_text = one_hot(clean_text,vocab_size)
		
	smax_length= 20
	#embedded representation
	embeded = pad_sequences([one_hot_text],padding='pre',maxlen=smax_length)
	return model.predict(embeded)


if __name__ == '__main__':
	app.run(debug=True)