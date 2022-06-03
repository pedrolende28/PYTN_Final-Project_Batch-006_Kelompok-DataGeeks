from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/rf_model2.pkl', 'rb'))
scaler_ = pickle.load(open('models/scaler_.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
   
    age = int(request.form['age'])
    anemia = int(request.form['anemia'])
    e_f = float(request.form['e_f'])
    s_sodium = float(request.form['s_sodium'])
    sex = int(request.form['sex'])
    time = int(request.form['time'])

    val = [age,anemia, e_f, s_sodium, sex, time]
    val = scaler_.transform([val])
    val_predict = model.predict(val)
    return render_template('predict.html', data=val_predict)

if __name__ == "__main__":
    app.run(debug=True)