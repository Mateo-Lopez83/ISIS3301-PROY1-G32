import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from DataModel import DataModel
from joblib import load
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model_pipeline = load("models/fakenews.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message":"hola"}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        df_data = pd.DataFrame([dataModel.dict()])
        df_columns = df_data.columns

        labels = model_pipeline.pipeline.predict(df_data[df_columns])
        probas = model_pipeline.pipeline.predict_proba(df_data[df_columns])

        results = []
        for label, proba in zip(labels, probas):
            max_prob = max(proba)  
            results.append({
                "label": int(label),
                "probability": float(max_prob)
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")), sep=";")
        if not all(col in df.columns for col in ["Titulo", "Descripcion"]):
            raise HTTPException(status_code=400, detail="Error en el formato del csv")

        df_data = df[["Titulo", "Descripcion"]]
        labels = model_pipeline.pipeline.predict(df_data)
        probas = model_pipeline.pipeline.predict_proba(df_data)

        predictions = []
        for label, proba in zip(labels, probas):
            max_prob = max(proba)
            predictions.append({
                "label": int(label),
                "probability": float(max_prob)
            })

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        metrics = model_pipeline.check()
        return {"metricas": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    global model_pipeline
    try:
        content = await file.read()
        new_data = pd.read_csv(StringIO(content.decode("utf-8")))
        if not all(col in new_data.columns for col in ["ID","Label","Titulo","Descripcion","Fecha"]):
            raise HTTPException(status_code=400, detail="Error en el formato del csv nuevo")
        model_pipeline.retrain(new_data)
        model_pipeline = load("models/fakenews.joblib")
        metrics = model_pipeline.check()
        return {"message": "Model retrained successfully",
                "metricas": metrics}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# probamos el API
