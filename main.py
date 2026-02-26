from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Make sure this is the correct path to your model file
model = joblib.load("model.pkl")

# Your input class remains clean with underscores
class ChocolateSale(BaseModel):
    Sales_Person: int
    Country: int
    Product: int
    Boxes_Shipped: int
    Day: int
    Month: int
    Year: int

@app.get("/")
def read_root():
    return{"message":"Model is ready!"}



@app.post("/predict")
def predict(data: ChocolateSale):
    # We remove the 0.0 because the model is expecting exactly 7 features.
    # The order must be: Sales Person, Country, Product, Boxes Shipped, Day, Month, Year
    raw_data = [
        data.Sales_Person,
        data.Country,
        data.Product,
        data.Boxes_Shipped,
        data.Day,
        data.Month,
        data.Year
    ]
    
    # Predict using the 7 features
    prediction = model.predict([raw_data])
    
    return {"prediction": round(float(prediction[0]), 2)}