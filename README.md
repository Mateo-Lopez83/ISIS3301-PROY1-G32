# Iniciar el API

Para iniciar este API se deben correr los siguentes comandos, para crear el ambiente virtual e instalar las librerias necesarias 

```
python -m venv venv

. venv/Scripts/activate

pip install -r 'requirements.txt'
```
Seguido de esto, es necesario generar el archivo ```.joblib``` para poder utilizar nuestro modelo. Es necesario correr el siguente comando, ya con el ambiente virtual activado

```
python api/runtime.py
```

Por ultimo, se corre el API

```
cd api

uvicorn main:app --reload
```

## Frontend

Esta API es consumida por el front, diseñado para interactuar con cada uno de los endpoints expuestos y ofrecer al usuario una buena experiencia verificando la veracidad de sus noticias

[Frontend Repo](https://github.com/ignchap27/ISIS3301-PROY1-G32-FRONT.git)
