import pandas as pd
import numpy as np
import unicodedata
import re
import joblib
import nltk
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import FunctionTransformer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
wpt = WordPunctTokenizer()
stemmer = SnowballStemmer("spanish")

class FakeNewsPipeline:
    class TextCleaner:
        @staticmethod
        def text_cleaning(doc):
            doc = FakeNewsPipeline.eliminar_acento(doc)
            doc = FakeNewsPipeline.eliminar_caracteres_esp(doc)
            doc = FakeNewsPipeline.convertir_minuscula(doc)
            tokens = FakeNewsPipeline.tokenizar(doc)
            tokens = FakeNewsPipeline.remove_stopwords(tokens)
            tokens = FakeNewsPipeline.stemming(tokens)
            return ' '.join(tokens)
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            return [self.text_cleaning(doc) for doc in X]
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._load_data()
        self._split_data()
        self._build_pipeline()
    
    @staticmethod
    def rellenar_nulos_texto(X):
        return X.fillna('')
    
    @staticmethod
    def eliminar_acento(doc):
        return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    @staticmethod
    def eliminar_caracteres_esp(doc):
        return re.sub(r'[^a-zA-ZñÑ\s]', '', doc)
    
    @staticmethod
    def convertir_minuscula(doc):
        return doc.lower().strip()
    
    @staticmethod
    def tokenizar(doc):
        return wpt.tokenize(doc)
    
    @staticmethod
    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]
    
    @staticmethod
    def stemming(tokens):
        return [stemmer.stem(word) for word in tokens]
    
    def _load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.X = self.df[['Titulo', 'Descripcion']]
        self.y = self.df['Label']
    
    def _split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
    
    def _build_pipeline(self):
        text_pipeline = Pipeline([
            ('fill_na', FunctionTransformer(self.rellenar_nulos_texto, validate=False)),
            ('cleaner', self.TextCleaner()),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
        ])

        preprocessor = ColumnTransformer([
            ('titulo', text_pipeline, 'Titulo'),
            ('descripcion', text_pipeline, 'Descripcion')
        ])

        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=70, random_state=42))
        ])
    
    def train(self):
        self.pipeline.fit(self.X_train, self.y_train)
    
    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_test)
        data = (classification_report(self.y_test, y_pred))
        return data
    
    def check(self):
        y_pred = self.pipeline.predict(self.X_test)
        data = classification_report(self.y_test, y_pred, output_dict=True) 
        return data
    
    def retrain(self, new_data: pd.DataFrame):
        self.df = pd.concat([self.df, new_data], ignore_index=True).drop_duplicates()
        self._split_data()
        self.train()
        self.save_model("models/fakenews.joblib")
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def save_model(self, filename):
        joblib.dump(self, filename)

    def predict_with_proba(self, X):
        pred = self.pipeline.predict(X)
        probas = self.pipeline.predict_proba(X)
        max_probas = probas.max(axis=1)  # Tomamos la probabilidad de la clase predicha
        return list(zip(pred, max_probas))

