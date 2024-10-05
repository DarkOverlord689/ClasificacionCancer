from app.models.prediction import PredictionInput, PredictionOutput
from src.load_and_predict import predicto, ImagePreprocessor

async def predict_image(file_path: str, prediction_input: PredictionInput) -> PredictionOutput:
    result = predicto(
        "ensemble",
        file_path,
        prediction_input.paciente.edad,
        prediction_input.paciente.sexo,
        prediction_input.diagnostico.localizacion
    )
    
    return PredictionOutput(
        predicted_class=result['predicted_class'],
        probabilities=result['probabilities']
    )
