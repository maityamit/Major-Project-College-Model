from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and encoders
clf = joblib.load('hair_condition_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class HairConditionInput(BaseModel):
    age: int
    gender: str
    family_history: str
    chronic_illness: str
    late_night: str
    sleep_disturbance: str
    water_quality: str
    hair_products: str
    anemia: str
    stress: str
    food_habit: str

# Define the prediction endpoint
@app.post("/predict")
def predict_hair_condition(input_data: HairConditionInput):

    feature_name_mapping = {
        "age": "What is your age ?",
        "gender": "What is your gender ?",
        "family_history": "Is there anyone in your family having a hair fall problem or a baldness issue?",
        "chronic_illness": "Did you face any type of chronic illness in the past?",
        "late_night": "Do you stay up late at night?",
        "sleep_disturbance": "Do you have any type of sleep disturbance?",
        "water_quality": "Do you think that in your area water is a reason behind hair fall problems?",
        "hair_products": "Do you use chemicals, hair gel, or color in your hair?",
        "anemia": "Do you have anemia?",
        "stress": "Do you have too much stress",
        "food_habit": "What is your food habit"
    }


    # Convert input data to a dictionary
    input_dict = {feature_name_mapping[key]: value for key, value in input_data.dict().items()}

    # Preprocess the input data
    for col, le in label_encoders.items():
        if col in input_dict:
            input_dict[col] = le.transform([input_dict[col]])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict the hair condition
    prediction = clf.predict(input_df)
    result = "Hair Fall Problem" if prediction[0] == 1 else "No Hair Fall Problem"

    return {"prediction": result}