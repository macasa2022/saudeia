import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = '7965f45d8c0e1f3bfa5a9a0e1f7d4a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7'  # ou 'Bissau@2021'

    MODEL_PATH = os.path.join(basedir, 'models', 'disease_model.pkl')
    VECTORIZER_PATH = os.path.join(basedir, 'models', 'vectorizer.pkl')
    SYMPTOMS_CSV = os.path.join(basedir, 'models', 'symptoms.csv')

    DEBUG = False