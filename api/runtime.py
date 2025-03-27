from pipline import FakeNewsPipeline
import pandas as pd

path = 'data/fake_news_spanish_limpio.csv'

pipeline = FakeNewsPipeline(path)
pipeline.train()
pipeline.save_model("prueba_modelo.joblib")
print('joblib creado cheveremenente')