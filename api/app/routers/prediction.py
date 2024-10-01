from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from app.db.crud import create_prediction, get_db, create_paciente, create_diagnostico, create_imagen
from app.models.prediction import PredictionOutput, PredictionCreate, PacienteCreate, DiagnosticoCreate, ImagenCreate, Paciente, Diagnostico, Imagen
from app.db.models import Diagnostico as DBDiagnostico, Paciente as DBPaciente, Imagen as DBImagen
from src.load_and_predict import predicto
import shutil
import os
import json
from typing import List

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    localization: str = Form(...),
    name: str = Form(...),
    identification: str = Form(...)
):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        patient_data = {
            "nombre": name,
            "numero_identificacion": identification,
            "edad": age,
            "sexo": sex,
        }

        result = predicto(
            "ensemble",
            temp_file,
            age,
            sex,
            localization
        )
        print(result['probabilities'])
        db: Session = next(get_db())
        paciente = create_paciente(db, PacienteCreate(**patient_data))
        
        diagnostico_data = {
            "localizacion": localization,
            "tipo_cancer": result['predicted_class'],
            "probabilidades": result['probabilities'],  
            "paciente_id": paciente.id
        }
        diagnostico = create_diagnostico(db, DiagnosticoCreate(**diagnostico_data))

        imagen_data = {
            "ruta_imagen": temp_file,
            "tipo_imagen": file.content_type,
            "diagnostico_id": diagnostico.id
        }
        imagen = create_imagen(db, ImagenCreate(**imagen_data))
        
        return PredictionOutput(
            paciente=Paciente.from_orm(paciente),
            diagnostico=Diagnostico.from_orm(diagnostico),
            imagen=Imagen.from_orm(imagen),
            predicted_class=result['predicted_class'],
            probabilities=result['probabilities']
        )
    finally:
        os.remove(temp_file)

@router.post("/create_prediction", response_model=PredictionOutput)
def create_prediction(prediction: PredictionCreate, db: Session = Depends(get_db)):
    try:
        paciente = create_paciente(db, prediction.paciente)
        
        diagnostico = create_diagnostico(db, DiagnosticoCreate(
            paciente_id=paciente.id,
            localizacion=prediction.diagnostico.localizacion,
            tipo_cancer=prediction.prediction_output.predicted_class,
            probabilidades=prediction.prediction_output.probabilities
        ))
        
        imagen = create_imagen(db, prediction.imagen)
        
        return PredictionOutput(
            paciente=Paciente.from_orm(paciente),
            diagnostico=Diagnostico.from_orm(diagnostico),
            imagen=Imagen.from_orm(imagen),
            predicted_class=prediction.prediction_output.predicted_class,
            probabilities=prediction.prediction_output.probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get_predictions", response_model=List[PredictionOutput])
def get_predictions(db: Session = Depends(get_db)):
    try:
        predictions = db.query(DBDiagnostico).join(DBPaciente).outerjoin(DBImagen).all()
        
        results = []
        for prediction in predictions:
            paciente = prediction.paciente
            imagen = prediction.imagenes[0] if prediction.imagenes else None
            
            results.append(PredictionOutput(
                paciente=Paciente.from_orm(paciente),
                diagnostico=Diagnostico.from_orm(prediction),
                imagen=Imagen.from_orm(imagen) if imagen else None,
                predicted_class=prediction.tipo_cancer,
                probabilities=prediction.probabilidades if prediction.probabilidades else {}
            ))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
