from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import os
def generate_pdf_report(patient_data, result, new_file_path, images_folder, image_names):
    logo_path='ConSlogan/Color.png'
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=1))

    # Añadir logo
    try:
        logo = Image(logo_path, width=2*inch, height=1*inch)
        story.append(logo)
    except:
        story.append(Paragraph("Logo no disponible", styles['Normal']))
    story.append(Spacer(1, 12))

    # Título
    story.append(Paragraph("Reporte de Análisis Dermatológico", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Información del paciente
    patient_info = [
        ['Nombre:', patient_data.get('Nombre', 'No especificado')],
        ['Identificación:', patient_data.get('Identificación', 'No especificado')],
        ['Edad:', patient_data.get('Edad', 'No especificado')],
        ['Sexo:', patient_data.get('Sexo', 'No especificado')],
        ['Localización:', patient_data.get('Localización', 'No especificado')]
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

    # Resultados
    story.append(Paragraph("Resultados del Análisis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Manejo de la clase predicha
    full_class_name = result.get('class', result.get('predicted_class', 'No especificado'))
    story.append(Paragraph(f"<b>Clase predicha:</b> {full_class_name}", styles['Normal']))
    story.append(Spacer(1, 6))
    
    # Manejo de probabilidades
    story.append(Paragraph("<b>Probabilidades:</b>", styles['Normal']))
    probabilities = result.get('probabilities', {})
    if isinstance(probabilities, dict):
        for class_name, probability in probabilities.items():
            story.append(Paragraph(f"- {class_name}: {probability:.4f}", styles['Normal']))
    else:
        story.append(Paragraph("Información de probabilidades no disponible", styles['Normal']))
    
    story.append(Spacer(1, 12))

    # Recomendación
    max_probability = max(probabilities.values()) if isinstance(probabilities, dict) and probabilities else 0
    if max_probability > 0.5:
        recommendation = "<font color='red'><b>Se recomienda consultar a un dermatólogo.</b></font>"
    else:
        recommendation = "<font color='green'><b>El riesgo parece bajo, pero consulte a un médico si tiene dudas.</b></font>"
    story.append(Paragraph(recommendation, styles['Normal']))

    # Espacio para imágenes
    story.append(Spacer(1, 12))
    story.append(Paragraph("Imágenes del Análisis", styles['Heading2']))
    story.append(Spacer(1, 6))
    # Añaade la imagen principal
    story.append(Image(new_file_path, width=6*inch, height=4*inch))
   
     # Añadir las imágenes desde la carpeta
    for image_name in image_names:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            story.append(Image(image_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 12))  # Espacio después de cada imagen
        else:
            story.append(Paragraph(f"Imagen no disponible: {image_name}", styles['Normal']))
    
    


    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


   
