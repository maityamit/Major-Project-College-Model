import pandas as pd
import joblib

# Load the trained model and encoders
clf = joblib.load('api/hair_condition_model.pkl')
label_encoders = joblib.load('api/label_encoders.pkl')

# Define the input features
input_data = {
    'What is your age ?': int(input("Enter your age: ")),
    'What is your gender ?': input("Enter your gender (Male/Female): "),
    'Is there anyone in your family having a hair fall problem or a baldness issue?': input("Family history of hair fall (Yes/No): "),
    'Did you face any type of chronic illness in the past?': input("Chronic illness in the past (Yes/No): "),
    'Do you stay up late at night?': input("Do you stay up late at night (Yes/No): "),
    'Do you have any type of sleep disturbance?': input("Do you have sleep disturbance (Yes/No): "),
    'Do you think that in your area water is a reason behind hair fall problems?': input("Is water quality a reason for hair fall (Yes/No): "),
    'Do you use chemicals, hair gel, or color in your hair?': input("Do you use chemicals or hair products (Yes/No): "),
    'Do you have anemia?': input("Do you have anemia (Yes/No): "),
    'Do you have too much stress': input("Do you have too much stress (Yes/No): "),
    'What is your food habit': input("Enter your food habit (Vegetarian/Non-Vegetarian): ")
}

# Preprocess the input data
for col, le in label_encoders.items():
    if col in input_data:
        input_data[col] = le.transform([input_data[col]])[0]

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict the hair condition
prediction = clf.predict(input_df)
print(f"Predicted Hair Condition: {'Hair Fall Problem' if prediction[0] == 1 else 'No Hair Fall Problem'}")