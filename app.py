# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Titanic_99.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Class = int(request.form['class'])
        Sex = int(request.form['sex'])
        Age = int(request.form['age'])
        Sib = int(request.form['sib'])
        Parch = int(request.form['parch'])
        Fare = int(request.form['fare'])
        Embarked = float(request.form['embarked'])
        
        Class = Class/3
        Age = Age/80
        Sib = Sib/8
        Fare = Fare/511.999

        x = np.array([[Class, Sex, Age, Sib, Parch, Fare, Embarked]])
        my_prediction = classifier.predict(x)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)