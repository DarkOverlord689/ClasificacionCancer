import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from reportlab.lib.units import inch

def generate_pdf(request_data):
    """
    Genera un PDF con el reporte del análisis dermatológico.
    
    Args:
        request_data: Objeto PredictionOutput con los datos del análisis
    """
    # Verificar que los directorios existan
    os.makedirs(request_data.patient_dir, exist_ok=True)

    # Verificar que la imagen original existe
    if not os.path.exists(request_data.original_image_path):
        raise FileNotFoundError("No se encontró la imagen original del análisis")

    # Generar el PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=1))

    # Logo
    logo_path = 'app\\services\\ConSlogan\\Color.png'
    try:
        logo = Image(logo_path, width=2*inch, height=1*inch)
        story.append(logo)
    except:
        story.append(Paragraph("Logo no disponible", styles['Normal']))
    story.append(Spacer(1, 12))

    # Título
    story.append(Paragraph("Reporte de Análisis Dermatológico", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Información del paciente y otros detalles
    patient_data = request_data.result.paciente
    diagnostic_data = request_data.result.diagnostico
    
    patient_info = [
        ['Nombre:', str(patient_data.nombre)],
        ['Identificación:', str(patient_data.numero_identificacion)],
        ['Edad:', str(patient_data.edad)],
        ['Sexo:', str(patient_data.sexo)],
        ['Localización:', str(diagnostic_data.localizacion)],
        ['Observación:', str(diagnostic_data.observacion)]
    ]

    patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # Resultados del análisis
    story.append(Paragraph("Resultados del Análisis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    predicted_class = request_data.result.predicted_class
    story.append(Paragraph(f"<b>Clase predicha:</b> {predicted_class}", styles['Normal']))
    story.append(Spacer(1, 6))

    # Probabilidades y Recomendación
    probabilities = request_data.result.probabilities
    story.append(Paragraph("<b>Probabilidades:</b>", styles['Normal']))
    
    # Convertir probabilidades a diccionario si es necesario
    if hasattr(probabilities, '__dict__'):
        prob_dict = probabilities.__dict__
    elif hasattr(probabilities, 'dict'):
        prob_dict = probabilities.dict()
    else:
        prob_dict = dict(probabilities)

    for class_name, probability in prob_dict.items():
        story.append(Paragraph(f"- {class_name}: {float(probability):.4f}", styles['Normal']))

    max_probability = max(prob_dict.values()) if prob_dict else 0
    recommendation = "<font color='red'><b>Se recomienda consultar a un dermatólogo.</b></font>" if max_probability > 0.5 else "<font color='green'><b>El riesgo parece bajo, pero consulte a un médico si tiene dudas.</b></font>"
    story.append(Paragraph(recommendation, styles['Normal']))

    # Imágenes
    story.append(Spacer(1, 12))
    story.append(Paragraph("Imagen Original", styles['Heading2']))
    story.append(Spacer(1, 6))

    try:
        story.append(Image(request_data.original_image_path, width=6*inch, height=4*inch))
    except Exception as e:
        story.append(Paragraph(f"No se pudo cargar la imagen original: {str(e)}", styles['Normal']))

    # Si hay imágenes de interpretación
    if hasattr(request_data, 'interpretation_images'):
        for img_path in request_data.interpretation_images:
            try:
                story.append(Image(img_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 6))
            except Exception as e:
                story.append(Paragraph(f"No se pudo cargar la imagen de interpretación: {str(e)}", styles['Normal']))

    # Construir el PDF
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()

    # Crear nombre del archivo
    filename = f"reporte_dermatologico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return pdf, filename
