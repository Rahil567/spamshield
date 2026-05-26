# 🔥 SpamShield — SMS Spam Detector

A **machine learning web application** that detects spam messages in real-time using TF-IDF vectorization and Logistic Regression — built with Python and Streamlit, no external API required.

---

## 🚀 Live Demo

> 🌐 [Live Demo](https://spamshield-mvsq.onrender.com)

---

## 📌 Project Overview

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using the SMS Spam Collection dataset with over **5,500 messages**.

| Component      | Details                              |
| -------------- | ------------------------------------ |
| **Model**      | Logistic Regression                  |
| **Vectorizer** | TF-IDF (unigrams + bigrams)          |
| **Dataset**    | SMS Spam Collection — 5,574 messages |
| **Accuracy**   | ~98%                                 |
| **Deployment** | Streamlit + Render                   |

---

## 🖼️ App Preview

### 🔍 Detect Tab — Single Message Analysis

Paste any SMS or email and get instant spam probability with a confidence bar.

![Detect Tab](D:\DELL\Desktop\projects\Detect_Tab.png)

### 📋 Bulk Scan Tab — Multi Message Scanner

Paste up to 50 messages at once and get a full results table with spam/ham summary.

![Bulk Scan](D:\DELL\Desktop\projects\Bulk _Scan.png)

### 📊 Model Insights Tab — Performance Metrics

View accuracy, precision, recall, F1 score, top spam indicator words, and full classification report.

![Model Insights](D:\DELL\Desktop\projects\Model_Insights.png)

---

## 🧠 What I Built

| Component         | Details                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| **Dataset**       | SMS Spam Collection — 5,574 labelled messages                           |
| **Cleaning**      | Lowercase, punctuation removal (numbers kept as spam signal)            |
| **Vectorization** | TF-IDF with unigrams + bigrams, `sublinear_tf=True`, stop-words removed |
| **Model**         | Logistic Regression with `class_weight='balanced'`                      |
| **Persistence**   | Model saved as `.pkl` — no retraining on every restart                  |
| **App**           | Streamlit — black UI, orange-red theme, 3-tab layout                    |

---

## 📊 Model Performance

**Test Accuracy: ~98%**

```
              precision    recall  f1-score

         Ham       0.99      0.99      0.99
        Spam       0.94      0.93      0.93

    accuracy                           0.98
   macro avg       0.96      0.96      0.96
weighted avg       0.98      0.98      0.98

```

---

## 🤔 Why Logistic Regression?

**Chosen because:**

- Trains in seconds on 5,500 messages
- `class_weight='balanced'` handles the natural ham/spam imbalance
- TF-IDF + bigrams captures phrases like "call now", "free prize" effectively
- Lightweight — runs on Render free tier without GPU
- Interpretable — coefficients directly show which words drive spam predictions

---

## 🔮 Future Roadmap

- [ ] Multilingual support — detect spam in Hindi, Gujarati
- [ ] Highlight which words triggered the spam label
- [ ] Flag borderline predictions with a confidence threshold
- [ ] Extend to full email body analysis
- [ ] User feedback to improve model over time

---

## ⚙️ How to Run Locally

**1. Clone the repo**

```
git clone https://github.com/Rahil567/spamshield.git
cd spamshield
```

**2. Install dependencies**

```
pip install -r requirements.txt
```

**3. Train the model** *(only once — saves .pkl files)*

```
python E_spam.py
```

**4. Launch the app**

```
streamlit run app.py
```

> Make sure `spam.csv` is in the same folder before running.

---

## 📁 Project Structure

```
spamshield/
│
├── app.py                 # Streamlit web application
├── E_spam.py              # ML pipeline — training, prediction, evaluation
├── spam.csv               # SMS Spam Collection dataset
├── spam_model.pkl         # Trained model (auto-generated)
├── spam_vectorizer.pkl    # TF-IDF vectorizer (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn

```

---

## 🌐 Deploy on Render

1. Push all files including `.pkl` files to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your repo
4. Set start command:

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

```

5. Click **Deploy**

> Commit `spam_model.pkl` and `spam_vectorizer.pkl` to avoid retraining on every deploy.

---

## 🔍 Key Learnings

- **Class imbalance** — dataset is ~87% ham, so `class_weight='balanced'` is critical to avoid the model ignoring spam
- **Bigrams matter** — phrases like "call now", "win free", "claim prize" are far stronger spam signals than single words
- **sublinear_tf** — dampens the effect of very frequent terms, improving generalization
- **Numbers as signal** — phone numbers and prize amounts are strong spam indicators, so punctuation is removed but digits are kept
- **Deployment constraints** — `.pkl` persistence means the app loads instantly without retraining on Render

---

## 👨‍💻 Author

**Rami Rahil Rohitbhai**  
JG University  
[LinkedIn](https://www.linkedin.com/in/rami-rahil-2a7538348) · [GitHub](https://github.com/Rahil567)

---

## 📄 Dataset

[SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

*Built with ❤️ using Python · Pandas · scikit-learn · Streamlit*  
*B.Tech AI & Data Science — Project*
