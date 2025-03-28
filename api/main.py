import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from DataModel import DataModel
from joblib import load
from io import StringIO

app = FastAPI()
model_pipeline = load("models/fakenews.joblib")

@app.get("/")
def read_root():
    return {"message":"hola"}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        print("im here")
        df_data = pd.DataFrame([dataModel.dict()])
        df_columns = df_data.columns

        #model = load("prueba_modelo.joblib")
        result = model_pipeline.predict(df_data[df_columns])         
        result_list = result.tolist() if hasattr(result, 'tolist') else result # se convierte a lista si es necesario
        return {"prediction": result_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        metrics = model_pipeline.check()
        return {"metricas": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batchpredict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        new_data = pd.read_csv(StringIO(content.decode("utf-8")))
        required_columns = ["ID", "Titulo", "Descripcion", "Fecha"]
        if not all(col in new_data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Error en el formato del CSV nuevo")
        df_for_prediction = new_data[["Titulo", "Descripcion"]]
        predictions = model_pipeline.predict(df_for_prediction)
        results = []
        for idx, pred in enumerate(predictions):
            results.append({
                "ID": new_data.iloc[idx]["ID"],
                "Titulo": new_data.iloc[idx]["Titulo"],
                "Descripcion": new_data.iloc[idx]["Descripcion"],
                "Fecha": new_data.iloc[idx]["Fecha"],
                "Prediction": pred
            })

        return {"predictions": results}

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
