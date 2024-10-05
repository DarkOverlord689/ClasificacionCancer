# app/models/prediction.py
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional

class PacienteBase(BaseModel):
    nombre: str
    numero_identificacion: int
    edad: int
    sexo: str

class PacienteCreate(PacienteBase):
    pass

class Paciente(PacienteBase):
    id: int
    fecha_registro: datetime

    class Config:
        orm_mode = True
        from_attributes = True

class DiagnosticoBase(BaseModel):
    localizacion: str
    tipo_cancer: str
    observacion: str  

class DiagnosticoCreate(DiagnosticoBase):
    paciente_id: int
    probabilidades: Optional[Dict[str, float]]

class Diagnostico(DiagnosticoBase):
    id: int
    paciente_id: int
    fecha_diagnostico: datetime
    probabilidades: Optional[Dict[str, float]]

    class Config:
        orm_mode = True
        from_attributes = True

class ImagenBase(BaseModel):
    ruta_imagen: str
    tipo_imagen: str

class ImagenCreate(ImagenBase):
    diagnostico_id: int

class Imagen(ImagenBase):
    id: int
    diagnostico_id: int
    fecha_imagen: datetime

    class Config:
        orm_mode = True
        from_attributes = True

class PredictionOutput(BaseModel):
    paciente: Paciente
    diagnostico: Diagnostico
    imagen: Optional[Imagen]
    predicted_class: str
    probabilities: Dict[str, float]

    class Config:
        orm_mode = True

class PredictionInput(BaseModel):
    age: int
    sex: str
    localization: str
    name: str
    identification: str
    observacion: str  
class PredictionCreate(BaseModel):
    paciente: PacienteCreate
    diagnostico: DiagnosticoCreate
    imagen: ImagenCreate
    prediction_output: PredictionOutput