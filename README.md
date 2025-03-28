# Iniciar el API

Para iniciar este API se deben correr los siguentes comandos, para crear el ambiente virtual e instalar las librerias necesarias 

```
python -m venv venv

. venv/Scripts/activate

pip install -r 'requirements.txt'

cd api

uvicorn main:app --reload
```
