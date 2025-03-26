from pydantic import BaseModel

class DataModel(BaseModel):
    Titulo: str
    Descripcion: str

def columns(self):
    return['Titulo', 'Descripcion']
