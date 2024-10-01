from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Dict
import uvicorn
import shutil
import os
from load_and_predict import predicto, ImagePreprocessor

app = FastAPI()

class PredictionInput(BaseModel):
    age: int
    sex: str
    localization: str
    name: str
    identification: str

class PredictionOutput(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]

@app.post("/predict", response_model=PredictionOutput)
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    localization: str = Form(...),
    name: str = Form(...),
    identification: str = Form(...)
):
    # Save the uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Prepare patient data
        patient_data = {
            "Nombre": name,
            "Identificación": identification,
            "Edad": age,
            "Sexo": sex,
            "categoria": "malignant",
            "Localización": localization
        }
        
        # Call the prediction function
        result = predicto(
            "ensemble",  # You might want to make this configurable
            temp_file,
            patient_data["Edad"],
            patient_data["Sexo"],
            patient_data["Localización"]
        )
        
        return PredictionOutput(
            predicted_class=result['predicted_class'],
            probabilities=result['probabilities']
        )
    finally:
        # Clean up the temporary file
        os.remove(temp_file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)