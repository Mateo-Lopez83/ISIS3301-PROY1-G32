import pandas as pd
from fastapi import FastAPI, HTTPException
from api.DataModel import DataModel
from joblib import load
from joblib import load

app = FastAPI()
model_pipeline = load("models/fakenews.joblib")

@app.get("/")
def read_root():
    return {"message":"hola"}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        df_data = pd.DataFrame([dataModel.dict()])
        df_columns = df_data.columns

        model = load("models/fakenews.joblib")
        result = model.predict(df_data[df_columns])         
        result_list = result.tolist() if hasattr(result, 'tolist') else result # se convierte a lista si es necesario
        return {"prediction": result_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        metrics = model_pipeline.evaluate()
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# probamos el API
