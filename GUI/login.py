import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class LoginWindow(QWidget):
    def __init__(self, on_successful_login):
        super().__init__()
        self.on_successful_login = on_successful_login
        self.setWindowTitle('Login - DermaDetect')
        self.setGeometry(100, 100, 400, 250)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Add logo
        self.logo_label = QLabel()
        self.logo_pixmap = QPixmap('ConSlogan/Color.png')
        self.logo_label.setPixmap(self.logo_pixmap.scaled(200, 100, Qt.AspectRatioMode.KeepAspectRatio))  # Adjust size as needed
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo_label)

        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)

        self.password_label = QLabel('Password:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton('Login')
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

    def apply_styles(self):
        # Set a background color for the main window
        self.setStyleSheet("background-color: #f5f5f5;")

        # Style the labels
        self.username_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.password_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")

        # Style the input fields
        self.username_input.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; padding: 8px; font-size: 14px; color: black;")
        self.password_input.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; padding: 8px; font-size: 14px; color: black;")

        # Style the login button
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == 'admin' and password == 'admin123':
            self.on_successful_login('admin')
        elif username == 'user' and password == 'user123':
            self.on_successful_login('user')
        else:
            QMessageBox.warning(self, 'Login Failed', 'Invalid username or password')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow(lambda user: print(f"{user} logged in"))
    window.show()
    sys.exit(app.exec())
