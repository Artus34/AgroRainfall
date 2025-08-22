import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Rainfall Prediction API",
    description="An API to predict annual rainfall based on monthly data and subdivision.",
    version="1.0.0"
)

# --- 2. Load Saved Artifacts ---
# These files are loaded once when the API starts.
try:
    with open('Best_Rainfall_Prediction_Model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('rainfall_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('rainfall_model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('rainfall_backend_mappings.pkl', 'rb') as f:
        backend_mappings = pickle.load(f)

    # Create a reverse mapping for easy lookups
    reverse_subdivision_map = {v: k for k, v in backend_mappings['subdivision_map'].items()}

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load a necessary artifact: {e}. Ensure all .pkl files are present.")

# --- 3. Define API Input/Output Models ---
class RainfallInput(BaseModel):
    SUBDIVISION_ID: int
    JAN: float = Field(..., example=10.5)
    FEB: float = Field(..., example=20.1)
    MAR: float = Field(..., example=30.2)
    APR: float = Field(..., example=40.3)
    MAY: float = Field(..., example=50.4)
    JUN: float = Field(..., example=150.5)
    JUL: float = Field(..., example=250.6)
    AUG: float = Field(..., example=240.7)
    SEP: float = Field(..., example=130.8)
    OCT: float = Field(..., example=60.9)
    NOV: float = Field(..., example=25.0)
    DEC: float = Field(..., example=15.1)

# --- 4. API Endpoints ---

@app.get("/", summary="API Root", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Rainfall Prediction API!"}


@app.get("/subdivisions", summary="Get Subdivisions", tags=["Categories"])
def get_subdivisions():
    """
    Provides the list of available subdivisions for frontend dropdowns.
    """
    return [{"id": v, "name": k} for k, v in backend_mappings['subdivision_map'].items()]


@app.post("/predict", summary="Predict Annual Rainfall", tags=["Prediction"])
def predict_rainfall(data: RainfallInput):
    """
    Receives monthly rainfall data and a subdivision ID to predict the total annual rainfall.
    """
    try:
        # Step A: Create a DataFrame with the correct structure, initialized to zeros
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Step B: Populate the monthly rainfall values
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        for month in months:
            input_df[month] = getattr(data, month)

        # Step C: Handle the one-hot encoded subdivision feature
        subdivision_name = reverse_subdivision_map.get(data.SUBDIVISION_ID)
        if not subdivision_name:
            raise HTTPException(status_code=400, detail="Invalid SUBDIVISION_ID provided.")

        # The first subdivision was dropped during encoding, so it's represented by all zeros.
        # We only need to set a column to 1 if it's NOT the first one.
        if subdivision_name != 'ANDAMAN & NICOBAR ISLANDS':
            col_name = f'SUBDIVISION_{subdivision_name}'
            if col_name in input_df.columns:
                input_df[col_name] = 1
            else:
                # This case should not happen if the mappings are correct
                raise HTTPException(status_code=500, detail=f"Internal error: Model column for {subdivision_name} not found.")

        # Step D: Scale the data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Step E: Make the prediction
        prediction = model.predict(input_scaled)[0]
        
        # Ensure the prediction is not negative
        final_prediction = max(0.0, prediction)

        return {"predicted_annual_rainfall_mm": f"{final_prediction:.2f}"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
