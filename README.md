# Iniciar el API

Para iniciar este API se deben correr los siguentes comandos, para activar el ambiente virtual y generar el archivo joblib que persiste nuestro modelo de fakenews

```
. venv/Scripts/activate
cd api
uvicorn main:app --reload
```
