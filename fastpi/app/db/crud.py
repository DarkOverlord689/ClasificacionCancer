from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models.prediction import PacienteCreate, DiagnosticoCreate, ImagenCreate, PredictionOutput
from app.db.models import Paciente, Diagnostico, Imagen


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_paciente(db: Session, paciente_data: PacienteCreate):
    paciente = Paciente(**paciente_data.dict()) 
    db.add(paciente)
    db.commit()
    db.refresh(paciente)
    return paciente

def create_diagnostico(db: Session, diagnostico_data: DiagnosticoCreate):
    diagnostico = Diagnostico(**diagnostico_data.dict())  
    db.add(diagnostico)
    db.commit()
    db.refresh(diagnostico)
    return diagnostico


def create_imagen(db: Session, imagen_data):
    imagen = Imagen(**imagen_data.dict()) 
    db.add(imagen)
    db.commit()
    db.refresh(imagen)
    return imagen

def create_prediction(db: Session, paciente: PacienteCreate, diagnostico: DiagnosticoCreate, prediction_output: PredictionOutput, ruta_imagen: str):
    # Crear paciente
    db_paciente = create_paciente(db, paciente)
    
    # Crear diagnóstico
    db_diagnostico = create_diagnostico(db, DiagnosticoCreate(
        **diagnostico.dict(),  # Cambiado aquí
        paciente_id=db_paciente.id,
        tipo_cancer=prediction_output.predicted_class
    ))
    
    # Crear imagen
    create_imagen(db, ImagenCreate(
        diagnostico_id=db_diagnostico.id,
        ruta_imagen=ruta_imagen,
        tipo_imagen="Original"
    ))
    
    return db_diagnostico