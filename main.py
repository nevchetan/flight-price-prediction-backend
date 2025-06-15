from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Use ["http://localhost:3000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("flight_price_model.pkl")

class FlightInput(BaseModel):
    Airline: int
    Source: int
    Destination: int
    Total_Stops: float
    Additional_Info: int
    date: int
    month: int
    arrival_hour: int
    arrival_min: int
    dept_hour: int
    dept_min: int
    duration_hour: int

@app.post("/predict")
def predict_price(data: FlightInput):
    input_data = np.array([[
        data.Airline, data.Source, data.Destination,
        data.Total_Stops, data.Additional_Info,
        data.date, data.month,
        data.arrival_hour, data.arrival_min,
        data.dept_hour, data.dept_min, data.duration_hour
    ]])
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0])}
