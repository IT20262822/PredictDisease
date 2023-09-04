from flask import Flask, render_template, request, redirect, url_for, session, make_response
from flask_cors import cross_origin, CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from flask_socketio import SocketIO, join_room, leave_room
from datetime import datetime
import socketio
from dbchat import save_message, get_messages
from pickle import load
from pandas import read_csv
from numpy import array
from csv import reader
from json import dumps
from google.cloud import dialogflow_v2
from google.cloud.dialogflow_v2.types import TextInput, QueryInput
from google.cloud.dialogflow_v2 import SessionsClient
from google.api_core.exceptions import InvalidArgument
from os import environ
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
app.secret_key = '09cd6cb8206a12b54a7ddb28566be757'
socketio = SocketIO(app, cors_allowed_origins="*")
bcrypt = Bcrypt(app)

cluster = MongoClient(
        "mongodb://localhost:27017/")
db = cluster['TrustyPet']
test_data = read_csv('ml/Testing.csv', sep=',')
test_data = test_data.drop('prognosis', axis=1)
symptoms = list(test_data.columns)
model = load(open('ml/new_logistic_regression_model.pkl', 'rb'))
db = cluster['TrustyPet']

fields = []
description = {}

with open('ml/disease_description.csv','r', encoding='utf-8') as csvfile:
    csvreader = reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        disease, desc = row
        description[disease] = desc
        
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

@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = ProcessRequest(req)
    res = dumps(res, indent=4)
    print(res)
    response = make_response(res)
    response.headers['Content-Type'] = 'application/json'
    return response


def ProcessRequest(req):
    collection = db['user_symptoms']
    result = req.get('queryResult')
    intent = result.get('intent').get('displayName')

    if intent == 'get_info':
        name = result.get('parameters').get('any')
        age = result.get('parameters').get('number')
        collection.insert_one({
            'name': name,
            'age': age,
            'symptoms': []
        })

        webhookresponse = 'Hey, what symptoms does your pet have? please enter one of the symptoms.(Ex. headache, vomiting etc.)'.format(
            name)

        return {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            webhookresponse
                        ]

                    }
                }
            ]
        }

    elif intent == 'get_symptom':
        name = result.get('outputContexts', [])[0].get('parameters').get('any')
        symptom = result.get('parameters').get('symptom')

        collection.find_one_and_update(
            {'name': name},
            {'$push': {'symptoms': symptom}}
        )

        webhookresponse = "Enter one more symptom beside {}. (Enter 'No' if not)".format(
            symptom)

        return {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            webhookresponse
                        ]
                    }
                }
            ]
        }
    elif intent == 'get_symptom - no':
        name = result.get('outputContexts', [])[1].get('parameters').get('any')
        user_symptoms = collection.find_one({'name': name})
        user_symptoms = list(user_symptoms.get('symptoms'))
        y = []
        for i in range(len(symptoms)):
            y.append(0)
        for i in range(len(user_symptoms)):
            y[symptoms.index(user_symptoms[i])] = 1
        disease = model.predict([array(y)])
        disease = disease[0]

        precaution_list = precautionDictionary[disease]

        webhookresponse = f"""Hey your pet might have {disease}.

        {description[disease]}
        
        """
        users = db['users']
        users.update_one({"name":name[1:]}, {"$set":{"disease":disease}})


        return {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            webhookresponse
                        ]
                    }
                }
            ]
        }


@app.route('/bot',methods=['POST'])
@cross_origin()
def bot():
    # symptom ="Fever"
    # data = {
    #     'SymptomOne': ['Fever', 'Cough2', 'Fatigue', 'Fever', 'Cough1'],
    #     # 'SymptomTwo': ['Cough3', 'Fever', 'Fatigue', 'Headache', 'Headache'],
    #     'Disease': ['Flu', 'Flu', 'Common Cold', 'Flu', 'Common Cold']
    # }

    # df = pd.DataFrame(data)
    # le = LabelEncoder()
    # for col in df.columns:
    #     df[col] = le.fit_transform(df[col])    
    #     X_train = df.drop('Disease',axis=1)
    #     y_train = df['Disease'].copy()
    #     print(df.dtypes)
    #     classifier = DecisionTreeClassifier()
    #     classifier.fit(X_train, y_train)
    # symptoms.append(le.transform([symptom])[0])
    # prediction = classifier.predict([symptoms])
    # print(f"Predicted Disease: {le.inverse_transform(prediction)[0]}")
    
    
    # response = make_response(res)
    # response.headers['Content-Type'] = 'application/json'
    # return response
    # age = user_data.get('age', 0)  # Replace with the actual JSON keys
    # symptoms = user_data.get('symptoms', '')
    # Convert symptoms to a numerical representation (dummy encoding)
    # Replace this with a proper encoding method in a real application
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from flask import Flask, request, jsonify
    from sklearn.preprocessing import MultiLabelBinarizer
    
    data = pd.read_csv('medical_data.csv')
    symptoms = [symptom.split(',') for symptom in data['Symptoms']]

    symptom_features = ['Cough', 'Fever', 'Headache']
    
    data['Symptoms'] = data['Symptoms'].apply(lambda x: x.split(','))

# Convert the Symptoms column to a list of lists for one-hot encoding
    symptoms = data['Symptoms'].tolist()
    
    # Initialize a MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(classes=symptom_features)
     
    
    mlb.fit(symptoms)
    symptoms_encoded = mlb.transform(symptoms)

    # Combine age and one-hot encoded symptoms
    X = pd.concat([data['Age'], pd.DataFrame(symptoms_encoded, columns=mlb.classes_)], axis=1)
    y = data['Disease']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a simple decision tree classifier (replace with a more complex model)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    user_data  = {"age": 30, "symptoms": "Fever"}
    age = user_data.get('age', 0)
    symptoms = user_data.get('symptoms', [])
    user_symptoms_encoded = mlb.transform([symptoms])

   
    # Combine age and one-hot encoded symptoms for prediction
    user_input = pd.DataFrame([[age] + list(user_symptoms_encoded[0])], columns=['Age'] + symptom_features)
    
    # Predict the disease using the trained model
    predicted_disease = classifier.predict(user_input)[0]


    print(jsonify({'predicted_disease': predicted_disease}))

if __name__ == '__main__':
    socketio.run(app, debug=True)
