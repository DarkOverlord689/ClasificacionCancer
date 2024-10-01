from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Diagnostico, Paciente

# Crear una sesión
engine = create_engine('sqlite:///./sql_app.db')
Session = sessionmaker(bind=engine)
session = Session()

# Consultar registros
registros = session.query(Diagnostico).all()

# Imprimir los registros
for registro in registros:
    print(registro)

# Cerrar la sesión
session.close()
