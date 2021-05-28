from flask import Flask, render_template, url_for, request
import pandas as pd, numpy as np
import pickle
   
# load the model from disk r 
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		me = request.form['message']
		try:
		    message = [float(x) for x in me.split(",")]
		except:
			return render_template('result.html',prediction = 1)
		if len(message)!=30:
			return render_template('result.html',prediction = 1)
		print(message)
		print(len(message))
		vect = np.array(message).reshape(1, -1)
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run()

