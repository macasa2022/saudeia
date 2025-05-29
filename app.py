from flask import Flask, render_template, request
import joblib
import pandas as pd
import spacy
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Carregar modelo e vetorizador
model = joblib.load(app.config['MODEL_PATH'])
vectorizer = joblib.load(app.config['VECTORIZER_PATH'])

# Carregar dados para descrição das doenças
df_diseases = pd.read_csv(app.config['SYMPTOMS_CSV']).set_index("doenca")

# Carregar modelo NLP
nlp = spacy.load("pt_core_news_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def chat():
    user_input = ""
    diagnosis = ""

    if request.method == "POST":
        user_input = request.form["symptoms"]
        cleaned = preprocess(user_input)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]

        # Obter recomendações
        try:
            sintomas = df_diseases.loc[prediction]["sintomas"]
        except:
            sintomas = "desconhecidos"

        diagnosis = f"""
        Com base nos seus sintomas, você pode estar infectado com <strong>{prediction}</strong>.
        Sintomas associados: {sintomas}.
        Recomendo que consulte um médico e realize exames específicos.
        """

    return render_template("index.html", diagnosis=diagnosis, user_input=user_input)

if __name__ == "__main__":
    app.run()