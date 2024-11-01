from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Body
from sqlalchemy.orm import Session
from app.db.crud import create_prediction, get_db, create_paciente, create_diagnostico, create_imagen
from app.models.prediction import PredictionOutput, PredictionCreate, PacienteCreate, DiagnosticoCreate, ImagenCreate, Paciente, Diagnostico, Imagen, PDFGenerationRequest
from app.db.models import Diagnostico as DBDiagnostico, Paciente as DBPaciente, Imagen as DBImagen
from load_and_predict import predicto
import shutil
import os
from typing import List, Dict
from fastapi.responses import FileResponse, Response
from app.services.pdf_services import generate_pdf
import logging
import traceback
from datetime import datetime
import glob

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    localization: str = Form(...),
    name: str = Form(...),
    identification: str = Form(...),
    observacion: str = Form(...)
):
    try:
        # Obtener rutas de almacenamiento
        timestamp = datetime.now()
        storage_paths = get_storage_paths(identification, timestamp)
        
        # Guardar la imagen en el directorio del paciente
        image_filename = f"original_{file.filename}"
        image_path = os.path.join(storage_paths["images_dir"], image_filename)
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Realizar predicción con la imagen guardada
        result = predicto(
            "ensemble",
            image_path,
            age,
            sex,
            localization,
            identification
        )
        
        # Crear registros en la base de datos
        db: Session = next(get_db())
        
        patient_data = {
            "nombre": name,
            "numero_identificacion": identification,
            "edad": age,
            "sexo": sex,
        }
        paciente = create_paciente(db, PacienteCreate(**patient_data))
        
        diagnostico_data = {
            "localizacion": localization,
            "tipo_cancer": result['predicted_class'],
            "probabilidades": result['probabilities'],
            "observacion": observacion,
            "paciente_id": paciente.id
        }
        diagnostico = create_diagnostico(db, DiagnosticoCreate(**diagnostico_data))
        
        imagen_data = {
            "ruta_imagen": image_path,  # Guardamos la ruta completa
            "tipo_imagen": file.content_type,
            "diagnostico_id": diagnostico.id
        }
        imagen = create_imagen(db, ImagenCreate(**imagen_data))
        
        # Preparar respuesta con rutas para el PDF
        prediction_output = PredictionOutput(
            paciente=Paciente.from_orm(paciente),
            diagnostico=Diagnostico.from_orm(diagnostico),
            imagen=Imagen.from_orm(imagen),
            predicted_class=result['predicted_class'],
            probabilities=result['probabilities']
        )
        
        # Agregar información de rutas a la respuesta
        prediction_output_dict = prediction_output.dict()
        prediction_output_dict.update({
            "storage_info": {
                "patient_dir": storage_paths["patient_dir"],
                "image_path": image_path,
                "diagnosis_date": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
        return prediction_output_dict
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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

@router.get("/get_predictions", response_model=List[PredictionOutput],
            summary="Recupera todos los registros de predicciones realizadas.",
            description="Obtiene todas las predicciones almacenadas en la base de datos.")
def get_predictions(db: Session = Depends(get_db)):
    """
    Retrieve all prediction records from the database.

    Returns a list of PredictionOutput objects containing patient data, diagnosis, image info, and prediction results.
    """
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
    
# Ruta donde se almacenan las imágenes generadas
IMAGENES_GENERADAS_PATH = "interpretations"

@router.get("/imagenes/")
async def obtener_imagenes():
    try:
        # Lista todos los archivos en la carpeta
        imagenes = os.listdir(IMAGENES_GENERADAS_PATH)
        
        # Filtra solo las imágenes (ajusta según tus necesidades)
        imagenes = [img for img in imagenes if img.endswith(('.png', '.jpg', '.jpeg'))]

        if not imagenes:
            raise HTTPException(status_code=404, detail="No se encontraron imágenes.")

        return {"imagenes": imagenes}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="La carpeta no fue encontrada.")
    except PermissionError:
        raise HTTPException(status_code=403, detail="No se tienen permisos para acceder a la carpeta.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Se produjo un error inesperado: {str(e)}")



@router.get("/imagenes/{nombre_imagen}")
async def servir_imagen(nombre_imagen: str):
    try:
        # Genera la ruta completa del archivo
        ruta_imagen = os.path.join(IMAGENES_GENERADAS_PATH, nombre_imagen)

        # Verifica si el archivo existe
        if not os.path.exists(ruta_imagen):
            raise HTTPException(status_code=404, detail="La imagen no fue encontrada.")

        # Devuelve la imagen como respuesta
        return FileResponse(ruta_imagen)
    
    except PermissionError:
        raise HTTPException(status_code=403, detail="No se tienen permisos para acceder a la imagen.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Se produjo un error inesperado: {str(e)}")
    

def get_storage_paths(identification: str, timestamp: datetime = None):
    if timestamp is None:
        timestamp = datetime.now()
    
    # Definir rutas base (ajusta según tu estructura)
    base_dir = "storage"  # Directorio base en el servidor
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Crear estructura de directorios
    patient_dir = os.path.join(base_dir, "patients", str(identification), date_str)
    images_dir = os.path.join(patient_dir, "images")
    reports_dir = os.path.join(patient_dir, "reports")
    
    # Asegurar que los directorios existan
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    return {
        "patient_dir": patient_dir,
        "images_dir": images_dir,
        "reports_dir": reports_dir
    }

@router.post("/generate-pdf",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Returns the generated PDF report"
        }
    })
async def generate_pdf_report_endpoint(
    request_data: PDFGenerationRequest = Body(...)
):
    try:
        logger.debug("Datos recibidos para generar PDF:")
        logger.debug(f"patient_dir: {request_data.result.paciente.numero_identificacion}")
        
        # Obtener rutas actualizadas
        storage_paths = get_storage_paths(
            request_data.result.paciente.numero_identificacion
        )
        
        # Actualizar rutas en los datos de solicitud
        request_data.patient_dir = storage_paths["patient_dir"]
        request_data.original_image_path = request_data.result.imagen.ruta_imagen
        
        # Validar que la imagen existe
        if not os.path.exists(request_data.original_image_path):
            raise FileNotFoundError(f"No se encuentra la imagen: {request_data.original_image_path}")
        
        # Generar PDF
        logger.info("Iniciando generación de PDF...")
        pdf, filename = generate_pdf(request_data)
        
        # Guardar una copia del PDF en el directorio de reportes
        pdf_path = os.path.join(storage_paths["reports_dir"], filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf)
        
        logger.info(f"PDF generado exitosamente: {filename}")
        
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
        )
        
    except Exception as e:
        logger.error("Error inesperado generando PDF:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdfs/", response_model=List[Dict[str, str]],
            summary="Recupera la lista de todos los PDFs almacenados",
            description="Obtiene una lista estructurada de todos los PDFs almacenados por paciente")
async def list_pdfs():
    try:
        base_dir = "storage/patients"
        if not os.path.exists(base_dir):
            raise HTTPException(status_code=404, detail="No se encontró el directorio de almacenamiento")

        pdf_files = []
        # Recorrer la estructura de directorios
        for patient_dir in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, patient_dir)
            if os.path.isdir(patient_path):
                # Buscar recursivamente todos los PDFs en los subdirectorios de reports
                pdf_pattern = os.path.join(patient_path, "**", "reports", "*.pdf")
                for pdf_path in glob.glob(pdf_pattern, recursive=True):
                    # Obtener información relevante del PDF
                    relative_path = os.path.relpath(pdf_path, base_dir)
                    filename = os.path.basename(pdf_path)
                    timestamp_dir = os.path.basename(os.path.dirname(os.path.dirname(pdf_path)))
                    
                    pdf_files.append({
                        "patient_id": patient_dir,
                        "filename": filename,
                        "timestamp": timestamp_dir,
                        "path": relative_path,
                    })

        return pdf_files

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar PDFs: {str(e)}")

@router.get("/pdfs/{patient_id}/{timestamp}/{filename}",
            summary="Descarga un PDF específico",
            description="Recupera un archivo PDF específico basado en el ID del paciente, timestamp y nombre del archivo")
async def get_pdf(patient_id: str, timestamp: str, filename: str):
    try:
        # Construir la ruta al archivo PDF
        pdf_path = os.path.join("storage", "patients", patient_id, timestamp, "reports", filename)
        
        # Verificar si el archivo existe
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF no encontrado")
            
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recuperar el PDF: {str(e)}")
    



from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


app = FastAPI(
    title="API de Predicción de Cáncer",
    description="API para la predicción de cáncer a partir de imágenes médicas",
    version="1.0.0"
)

# Incluye tu enrutador con un prefijo opcional y una etiqueta
app.include_router(router, prefix="/api", tags=["Predicciones de Cáncer"])

# Definir el esquema de OpenAPI personalizado (opcional)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API de Predicción de Cáncer",
        version="1.0.0",
        description="Esta API proporciona predicciones de cáncer basadas en imágenes médicas.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
