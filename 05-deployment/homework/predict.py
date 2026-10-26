import pickle

from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="customer-churn-prediction_homework")


class CustomerData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(customer: CustomerData):
    
    customer_dict = customer.model_dump()
    
    prob = float(model.predict_proba([customer_dict])[0, 1])

    return {"churn_probability": prob}