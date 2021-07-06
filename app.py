from flask import Flask, render_template, url_for, request
import numpy as np
import pickle
import os
# load the model from disk r 
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
path=os.getcwd()
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		me = request.form['message']
		me=me.rstrip()
		try:
		    message = [float(x) for x in me.split(",")]
		except:
			return render_template('result.html',prediction = 2)
		if len(message)!=30:
			return render_template('result.html',prediction = 2)
		print(message)
		print(len(message))
		vect = np.array(message).reshape(1, -1)
		my_prediction = clf.predict(vect)
		if my_prediction == 1:
			f1=open('fraudvalues.csv','a')
			s="\n"+me+",1"
			f1.write(s)
			f1.close()
		else:
			f1=open('validvalues.csv','a')
			s="\n"+me+",0"
			f1.write(s)
			f1.close()
	return render_template('result.html',prediction = my_prediction)
@app.route('/data')
def data():
	return render_template('csv.html')
if __name__ == "__main__":
    app.run()

