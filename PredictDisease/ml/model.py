from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd



train_data = pd.read_csv('Training.csv', sep=',')
test_data = pd.read_csv('Testing.csv', sep=',')

# Create a sample dataset
data = {
    'Symptom1': ['Fever', 'Cough', 'Fatigue', 'Fever', 'Cough'],
    'Symptom2': ['Cough', 'Fever', 'Fatigue', 'Headache', 'Headache'],
    'Disease': ['Flu', 'Flu', 'Common Cold', 'Flu', 'Common Cold']
}

df = pd.DataFrame(data)

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])
    
X_train = train_data.drop('prognosis',axis=1)
y_train = train_data['prognosis'].copy()
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis'].copy()

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
