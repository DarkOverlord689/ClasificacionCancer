#Maneja la visualización de datos y métricas.
import pandas as pd
from PyQt6.QtGui import QStandardItemModel, QStandardItem

def display_metrics(table_view, df):
    model = QStandardItemModel(len(df.index), len(df.columns))
    model.setHorizontalHeaderLabels(df.columns)

    for row in range(len(df.index)):
        for col in range(len(df.columns)):
            item = QStandardItem(str(df.iloc[row, col]))
            model.setItem(row, col, item)

    table_view.setModel(model)
    table_view.resizeColumnsToContents()

def plot_roc_curve(canvas):
    ax = canvas.figure.subplots()
    ax.clear()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Línea base')
    ax.plot([0, 0.2, 0.5, 0.8, 1], [0, 0.4, 0.7, 0.9, 1], label='Modelo')
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title('Curva ROC')
    ax.legend()
    canvas.draw()