from flask import Flask,render_template, request,url_for
from sys import prefix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

from werkzeug.utils import redirect
model = pickle.load(open('heartdiseaseModel.pkl', 'rb'))
app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
     if request.method == 'POST':
         age = request.form['age']
         sex = request.form['sex']
         cp = request.form['cp']
         trestbps = request.form['trestbps']
         chol = request.form['chol']
         fbs =request.form['fbs']
         restecg = request.form['restecg']
         thalach = request.form['thalach']
         exang = request.form['exang']
         oldpeak = request.form['oldpeak']
         slope = request.form['slope']
         ca = request.form['ca']
         thal =request.form['thal']
         prediction = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
         result = {'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}
         if(prediction == 0):
          
             return render_template('predect.html', prediction_text = "The person have No heart disease",result=result) 
         else:
        
            return render_template('predect1.html', prediction_text = "The person have heart disease",result=result)
         


        

# @app.route("/<usr>")
# def user(usr):
#     return f"<h1>{usr}<h1>"



if __name__ == "__main__":
    app.run(debug=True)
