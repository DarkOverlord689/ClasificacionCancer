
import sys
from PyQt6.QtWidgets import QApplication
from melanoma_detector import MelanomaDetector

if __name__ == '__main__':
    """
    Este script es la función principal y crea una aplicación PyQt para mostrar la interfacce gráfica
    """

    # Inicia la aplicación PyQt
    app = QApplication(sys.argv)

    # Crea una instancia de MelanomaDetector el cual es la función donde están las funcionalidades
    ex = MelanomaDetector()

    # Muestra la ventana
    ex.show()

    # Sale  de la aplicación cuando se cierre la ventan
    sys.exit(app.exec())
