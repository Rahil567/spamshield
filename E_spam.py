# SPAM CLASSIFIER MODULE

import pickle
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)


# CONSTANTS

DATA_PATH       = "spam.csv"
MODEL_PATH      = "spam_model.pkl"
VECTORIZER_PATH = "spam_vectorizer.pkl"


# 1. LOAD & PREPARE DATA 

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df.dropna(inplace=True)
    return df


# 2. TEXT CLEANING 

def clean_text(text: str) -> str:
    """Lowercase + remove punctuation. Numbers kept (spam signal)."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# 3. TRAIN PIPELINE

def train(data_path: str = DATA_PATH) -> dict:
    """
    Full training pipeline. Returns a metrics dict.
    Saves model + vectorizer to disk.
    """
    df = load_data(data_path)
    df["cleaned"] = df["text"].apply(clean_text)

    print(f"\nTotal samples : {len(df)}")
    print(f"Spam          : {df['label'].sum()}")
    print(f"Ham           : {(df['label'] == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF: unigrams + bigrams
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True          # dampens very high term frequencies
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Logistic Regression with class-weight balancing
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall"   : round(recall_score(y_test, y_pred)    * 100, 2),
        "f1"       : round(f1_score(y_test, y_pred)        * 100, 2),
        "report"   : classification_report(y_test, y_pred, target_names=["Ham", "Spam"]),
        "cm"       : confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n===== MODEL PERFORMANCE =====")
    print(metrics["report"])
    print("Confusion Matrix:", metrics["cm"])

    # Persist
    with open(MODEL_PATH, "wb") as f:      pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)
    print(f"\nSaved → {MODEL_PATH}, {VECTORIZER_PATH}")

    return metrics


# 4. LAZY LOAD

_model      = None
_vectorizer = None

def _load():
    global _model, _vectorizer
    if _model is None:
        import os
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, "rb") as f:      _model      = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f: _vectorizer = pickle.load(f)
        else:
            train()
            _load()


# 5. PUBLIC API 
def predict_spam(text: str) -> tuple[int, float]:
    """Returns (prediction, spam_probability). 1 = Spam, 0 = Ham."""
    _load()
    vec  = _vectorizer.transform([clean_text(text)])
    pred = int(_model.predict(vec)[0])
    prob = float(_model.predict_proba(vec)[0][1])
    return pred, round(prob, 4)


def top_spam_features(n: int = 10) -> list[tuple[str, float]]:
    """Top n words/bigrams most associated with spam."""
    _load()
    names  = _vectorizer.get_feature_names_out()
    coefs  = _model.coef_[0]
    idx    = coefs.argsort()[-n:][::-1]
    return [(names[i], round(float(coefs[i]), 3)) for i in idx]


def get_model_metrics() -> dict:
    """Re-evaluate on a fresh split and return metrics dict."""
    return train()


# 6. CLI SELF-TEST 

if __name__ == "__main__":
    train()

    test_msg = "Congratulations! You've won a FREE £1000 gift card. Claim now: http://win.xyz"
    pred, prob = predict_spam(test_msg)
    print(f"\nTest → {'SPAM' if pred else 'HAM'}  ({prob*100:.1f}% spam)")