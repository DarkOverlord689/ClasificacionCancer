from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import os
def generate_pdf_report(result, new_file_path, images_folder, image_names):
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
    patient_data = result['paciente']
    diagnostic_data = result['diagnostico']
    patient_info = [
        ['Nombre:', patient_data.get('nombre', 'No especificado')],
        ['Identificación:', str(patient_data.get('numero_identificacion', 'No especificado'))],
        ['Edad:', str(patient_data.get('edad', 'No especificado'))],
        ['Sexo:', patient_data.get('sexo', 'No especificado')],
        ['Localización:', diagnostic_data.get('localizacion', 'No especificado')],
        ['Observación:', diagnostic_data.get('observacion', 'No especificado')]
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
    
    # Clase predicha
    full_class_name = result.get('predicted_class', 'No especificado')
    story.append(Paragraph(f"<b>Clase predicha:</b> {full_class_name}", styles['Normal']))
    story.append(Spacer(1, 6))
    
    # Probabilidades
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
    
    # Añade la imagen principal
    if os.path.exists(new_file_path):
        story.append(Image(new_file_path, width=6*inch, height=4*inch))
    else:
        story.append(Paragraph("Imagen principal no disponible", styles['Normal']))
    
    # Crear lista para las imágenes y nombres
    image_elements = []
    for image_name in image_names:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            img = Image(image_path, width=2.4 * inch, height=2.4 * inch)
            # Crear una tabla para cada imagen y su nombre
            image_table = Table([[img], [Paragraph(image_name, styles['Normal'])]], 
                                colWidths=[2.5 * inch], 
                                rowHeights=[2.5 * inch, 0.5 * inch])
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            image_elements.append(image_table)
        else:
            # Manejo de imagen no disponible
            image_elements.append(Paragraph(f"Imagen no disponible: {image_name}", styles['Normal']))

    # Crear tabla con dos columnas para las imágenes solo si hay elementos
    if image_elements:
        num_rows = (len(image_elements) + 1) // 2  # Redondear hacia arriba
        image_grid = []
        for i in range(0, len(image_elements), 2):
            row = image_elements[i:i+2]
            if len(row) < 2:
                row.append("")  # Añadir una celda vacía si es necesario
            image_grid.append(row)

        image_table = Table(image_grid, colWidths=[3 * inch, 3 * inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        # Añadir la tabla de imágenes a la historia
        story.append(image_table)
    else:
        story.append(Paragraph("No hay imágenes de comparación disponibles", styles['Normal']))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
