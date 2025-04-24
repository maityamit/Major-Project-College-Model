import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your dataset's path

# Drop unnecessary columns (e.g., Timestamp, Name)
df = df.drop(['Timestamp', 'What is your name ?'], axis=1)

# Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target
X = df.drop('Do you have hair fall problem ?', axis=1)
y = df['Do you have hair fall problem ?']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and encoders if needed
import joblib
joblib.dump(clf, 'api/hair_condition_model.pkl')
joblib.dump(label_encoders, 'api/label_encoders.pkl')