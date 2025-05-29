import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import spacy

# Carregar modelo NLP
nlp = spacy.load("pt_core_news_sm")

# Carregar dados
df = pd.read_csv("symptoms.csv")
df['sintomas'] = df['sintomas'].str.replace(',', '').str.lower()

# Vetorizar sintomas
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['sintomas'])
y = df['doenca']

# Treinar modelo
model = MultinomialNB()
model.fit(X, y)

# Salvar modelo e vetorizador
joblib.dump(model, 'disease_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Modelo treinado e salvo!")