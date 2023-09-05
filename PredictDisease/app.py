from flask_socketio import SocketIO
from flask import Flask, render_template, request, redirect, url_for, session, make_response
from flask_cors import cross_origin, CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from numpy import array
from json import dumps
from os import environ
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
from sklearn.preprocessing import MultiLabelBinarizer


app = Flask(__name__)
app.secret_key = '09cd6cb8206a12b54a7ddb28566be757'
socketio = SocketIO(app, cors_allowed_origins="*")
bcrypt = Bcrypt(app)

cluster = MongoClient(
        "mongodb://localhost:27017/")
db = cluster['TrustyPet']
db = cluster['TrustyPet']

fields = []
description = {}

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/bot')
@cross_origin()
def chat_bot():
    if "email" in session:
        return render_template('bot.html')
    else:
        return redirect(url_for('login', next='/bot'))        

@app.route('/signup', methods=['POST'])
@cross_origin()
def sign_up():
    collection = db['users']
    form_data = request.form
    pw_hash =  bcrypt.generate_password_hash(form_data['password']).decode('utf-8')
    result = collection.find_one({'email': form_data['email']})
    if result == None:
        id = collection.insert_one({
            'name': form_data['name'],
            'email': form_data['email'],
            'password': pw_hash
        }).inserted_id
        response = ''
        if id == '':
            response = 'failed'
        else:
            response = 'success'
    else:
        response = 'failed'
    return render_template('signup.html', response=response)


@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        collection = db['users']
        form_data = request.form
        next_url = request.form.get('next')
        query_data = {
            "email": form_data['email']
        }
        result = collection.find_one(query_data)
        response = ''
        if result == None:
            response = 'failed'
        else:
            pw = form_data['password']
            if bcrypt.check_password_hash(result['password'], pw) == True:
                response = 'succeeded'
                session['email'] = form_data['email']
                session['name'] = result['name']
                
            else:
                response = 'failed'
        
        if response == 'failed':
            return render_template('login.html', response=response)
        else:
            if next_url:
                return redirect(next_url)
            else:
                return redirect(url_for('chat_bot'))
    else:
        if "email" in session:
            return redirect(url_for("chat_bot"))
        else:
            return render_template('login.html')
        
@app.route('/dashboard')
@cross_origin()
def dashboard():
    if "email" in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
@cross_origin()
def logout():
    if 'email' in session:
        session.pop("email", None)
        session.pop("username", None)
        
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))

@app.route('/bot',methods=['POST'])
@cross_origin()
def bot():
   
    # Load the dataset
    data = pd.read_csv('medical_data.csv')
    symptom_features = list(set(data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].values.ravel()))

    # Convert the Symptoms column to a list of lists for one-hot encoding
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].apply(lambda x: x.str.split(','))
   
   # Flatten the symptom lists
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].apply(lambda x: [item for sublist in x for item in sublist])
    
    symptoms = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].values.tolist()
   
    # Initialize a MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(classes=symptom_features)
    mlb.fit(symptoms)
    symptoms_encoded = mlb.transform(symptoms)

    # One-hot encoded symptoms
    X = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)
    y = data['Disease']

    # Create a decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X, y)


    user_data = request.json  
    # symptoms_input = user_data.get('symptoms', [])
    symptoms_input = user_data.get('symptoms', "").split(',')
    print(user_data)
    
    # Ensure that there are four symptoms, and if not, handle accordingly
    if len(symptoms_input) != 4:
        return jsonify({'error': 'Please provide four symptoms separated by commas.'}), 400
    
    # Encoding the user's symptoms using the same MultiLabelBinarizer
    user_symptoms_encoded = mlb.transform([symptoms_input])

    # One-hot encoded symptoms for prediction
    user_input = pd.DataFrame([list(user_symptoms_encoded[0])])

    # Predicting the disease using the trained model
    predicted_disease = classifier.predict(user_input)
    print(predicted_disease)
    return jsonify({'predicted_disease': predicted_disease[0]})

if __name__ == '__main__':
    app.run(debug=True)



