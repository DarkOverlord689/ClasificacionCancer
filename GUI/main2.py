import sys
from PyQt6.QtWidgets import QApplication
from login import LoginWindow
from melanoma_detector import MelanomaDetector



    """
    Este script es la función principal y crea una aplicación PyQt para mostrar la interfacce gráfica
    """

class App:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.login_window = LoginWindow(self.show_main_window)
        self.main_window = None

    def show_login(self):
        self.login_window.show()

    def show_main_window(self, user_type):
        self.main_window = MelanomaDetector(user_type)
        self.main_window.show()
        self.login_window.close()

    def run(self):
        self.show_login()
        sys.exit(self.app.exec())

if __name__ == '__main__':
    App().run()
