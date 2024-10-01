# main.py
from fastapi import FastAPI, Depends
from app.routers import prediction
from app.db.database import engine, Base
from app.db.crud import get_db


app = FastAPI()

# Crear tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Incluir rutas
app.include_router(prediction.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)