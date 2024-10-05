# utils.py
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATIENT_DATA_DIR = os.path.join(BASE_DIR, 'datos_paciente')
INTERPRETATIONS_DIR = os.path.join(BASE_DIR, 'interpretations')

def get_patient_directory(patient_id, diagnosis_date):
    patient_dir = os.path.join(PATIENT_DATA_DIR, str(patient_id))
    diagnosis_dir = os.path.join(patient_dir, diagnosis_date.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(diagnosis_dir, exist_ok=True)
    return diagnosis_dir

def get_interpretation_directory(patient_id, diagnosis_date):
    interpretation_dir = os.path.join(INTERPRETATIONS_DIR, str(patient_id), diagnosis_date.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(interpretation_dir, exist_ok=True)
    return interpretation_dir