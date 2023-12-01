import pandas as pd
import nltk
import re
import numpy as np
import string
from collections import Counter
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text(text) -> str:
    """
    Prétraite un texte en effectuant des tâches courantes de TALN telles que la suppression HTML,
    la suppression de la ponctuation, la tokenisation, la conversion en minuscules,
    la suppression des stopwords, la lemmatisation et le nettoyage des données.

    Paramètres:
    text (str): Le texte d'entrée à prétraiter.

    Renvois:
    str: Le texte traité après l'application de toutes les tâches de TALN.
    """
    # Initialiser un lemmatiseur WordNet pour obtenir les formes de base des mots et créer un ensemble de stopwords en anglais
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words("english")

    # Supprimer HTML
    # Solution utilisant regex provient de https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string.
    CLEANR = re.compile("<.*?>")
    text = re.sub(CLEANR, "", text)

    # Suppression de la ponctuation : Éliminer les signes de ponctuation. Se fait plus tard, avec les tokens!
    text = re.sub(r'[^\w\d\s\']+', ' ', text)
    # text = [s for s in text if s not in string.punctuation]
    # text = " ".join(text)

    # Tokenisation : Diviser le texte en mots
    tokens = word_tokenize(text)

    # Minuscules : Convertir les mots en minuscules
    tokens = [token.lower() for token in tokens]

    # Suppression des stopwords : Supprimer les stopwords courants
    tokens = [word for word in tokens if word not in stopwords_english and word not in string.punctuation]

    # Lemmatisation : Appliquer la lemmatisation pour réduire les mots à leur forme de base
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Nettoyage des données : Supprimer les jetons vides et effectuer tout nettoyage supplémentaire
    tokens = [token for token in tokens if token != "" and token != " "]

    # Joindre les jetons nettoyés pour former un texte traité
    text = " ".join(tokens)

    return text


def review_lengths(df) -> pd.Series:
    """
    Calculer le nombre de mots pour chaque élément dans une colonne d'un DataFrame pandas.

    Paramètres:
    - df (pd.Series): Une série pandas contenant les critiques nettoyées.

    Renvois:
    - pd.Series: Une série Pandas contenant le décompte de mots pour chaque élément de la série d'entrée.
    """

    # Diviser le texte en mots et calculer le nombre de mots.
    def n_word(text: str) -> int:
        return len(word_tokenize(text))

    res = df.map(n_word)
    return res


def word_frequency(df) -> pd.Series:
    """
    Calculer la fréquence des mots pour une colonne d'un DataFrame pandas.

    Paramètres:
    - df (pd.Series): Une série pandas contenant les critiques nettoyées.

    Renvois:
    pd.Series:
        Une série Pandas contenant les fréquences des mots pour chaque mot de la série d'entrée.
    """
    # Obtenir les fréquences de mots uniques et les renvoyer sous forme de pd.Series ordonnée par ordre décroissant.
    # Cela vous aidera à représenter graphiquement les 20 mots les plus fréquents et les 20 mots les moins fréquents.

    c = Counter()

    def count_word(text: str):
        words = word_tokenize(text)
        c.update(words)

    df.map(count_word)
    res = pd.Series(dict(c)).sort_values(ascending = False)
    return res


def encode_sentiment(df, sentiment_column="sentiment") -> pd.DataFrame:
    """
    Encoder la colonne de sentiment d'un DataFrame en valeurs numériques.

    Paramètres:
    - df (pd.DataFrame): Le DataFrame contenant la colonne de sentiment.
    - sentiment_column (str): Le nom de la colonne de sentiment. Par défaut, c'est 'sentiment'.

    Renvois:
    - pd.DataFrame: Un nouveau DataFrame avec la colonne de sentiment encodée en valeurs numériques.
    """
    df = df.copy()
    df[sentiment_column] = df[sentiment_column].map(
        lambda x: 1 if x == "positive" else 0
    )
    # Encoder nos étiquettes cibles en étiquettes numériques.

    return df


def explain_instance(
    tfidf_vectorizer, naive_bayes_classifier, X_test, idx, num_features=10
):
    """
    Expliquer une instance de texte en utilisant LIME (Local Interpretable Model-agnostic Explanations).

    Paramètres:
    - tfidf_vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Le vectoriseur TF-IDF entraîné.
    - naive_bayes_classifier (sklearn.naive_bayes.MultinomialNB): Le classificateur Naive Bayes entraîné.
    - X_test (pandas.core.series.Series): La liste des instances de texte à expliquer.
    - idx (int): L'index de l'instance à expliquer.
    - num_features (int, optionnel): Le nombre de caractéristiques (mots) à inclure dans l'explication. Par défaut, c'est 10.

    Renvois:
    - lime.lime_text.LimeTextExplainer: L'objet d'explication contenant des informations sur l'explication de l'instance.
    - float: La probabilité que l'instance soit classée comme 'positive' arrondie à 4 chiffres après la virgule.
    """

    # Créer un pipeline avec le vectoriseur et le modèle entraînés
    #
    X_test = X_test.copy()
    X_test = tfidf_vectorizer.transform(X_test[idx])
    print(X_test)
    pipe = make_pipeline(naive_bayes_classifier)
    # pipe = Pipeline(
    #     [
    #         ("vect", tfidf_vectorizer),
    #         ("clf", naive_bayes_classifier)
    #     ]
    # )
    # Spécifier les noms de classe
    classes = {1: "positive", 0: "negative"}
    # Créer un LimeTextExplainer
    exp = LimeTextExplainer(class_names=classes)
    # Expliquer l'instance à l'index spécifié
    explanation = exp.explain_instance(
        X_test[idx], pipe.predict_proba, num_features=num_features
    )
    # # Calculer la probabilité que l'instance soit classée comme 'positive'. Arrondir le résultat à 4 chiffres après la virgule.
    proba = round(pipe.predict_proba(X_test[idx]), 4)

    return explanation, proba
