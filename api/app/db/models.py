from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class Paciente(Base):
    __tablename__ = "pacientes"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    numero_identificacion = Column(Integer)
    edad = Column(Integer)
    sexo = Column(String(10))
    fecha_registro = Column(DateTime, default=datetime.utcnow)

    diagnosticos = relationship("Diagnostico", back_populates="paciente")

    def __repr__(self):
        return f"<Paciente(nombre='{self.Nombre}', identificacion='{self.Identificacion}', edad={self.Edad}, sexo='{self.Sexo}')>"

class Diagnostico(Base):
    __tablename__ = "diagnosticos"

    id = Column(Integer, primary_key=True, index=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    localizacion = Column(String(100))
    tipo_cancer = Column(String(50))
    probabilidades = Column(JSON)  # New column to store probabilities as JSON
    observacion = Column(String(500))
    fecha_diagnostico = Column(DateTime, default=datetime.utcnow)

    paciente = relationship("Paciente", back_populates="diagnosticos")
    imagenes = relationship("Imagen", back_populates="diagnostico")

    def __repr__(self):
        return (f"<Diagnostico(id={self.id}, paciente_id={self.paciente_id}, "
                f"localizacion='{self.localizacion}', tipo_cancer='{self.tipo_cancer}', "
                f"probabilidades={self.probabilidades}, "
                f"observacion='{self.observacion}', "
                f"fecha_diagnostico='{self.fecha_diagnostico}')>")

class Imagen(Base):
    __tablename__ = "imagenes"

    id = Column(Integer, primary_key=True, index=True)
    diagnostico_id = Column(Integer, ForeignKey("diagnosticos.id"))
    ruta_imagen = Column(String(255))
    tipo_imagen = Column(String(50))
    fecha_imagen = Column(DateTime, default=datetime.utcnow)

    diagnostico = relationship("Diagnostico", back_populates="imagenes")

    def __repr__(self):
        return (f"<Imagen(id={self.id}, diagnostico_id={self.diagnostico_id}, "
                f"ruta_imagen='{self.ruta_imagen}', tipo_imagen='{self.tipo_imagen}', "
                f"fecha_imagen='{self.fecha_imagen}')>")
