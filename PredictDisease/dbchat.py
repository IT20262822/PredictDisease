from pymongo import MongoClient, DESCENDING
from datetime import datetime

cluster = MongoClient(
        "mongodb://localhost:27017/")
db = cluster['Trustypet']

messages_collection = db.get_collection('messages')

def save_message(text, sender, userType):
    messages_collection.insert_one({
        'text':text,
        'sender':sender,
        'userType': userType,
        'created_at':datetime.now()
        })
    
def get_messages():
    messages = list(messages_collection.find({}).sort('_id', DESCENDING))
    for message in messages:
        message['created_at'] = message['created_at'].strftime("%d %b, %H:%M")
    return messages[::-1]