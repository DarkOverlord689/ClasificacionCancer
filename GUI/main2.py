import sys
from PyQt6.QtWidgets import QApplication
from melanoma_detector import MelanomaDetector

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MelanomaDetector()
    ex.show()
    sys.exit(app.exec())