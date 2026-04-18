# 🔍 Fake News Detector

A machine learning model that classifies news articles as **FAKE** or **REAL** 
with 91.5% accuracy using TF-IDF vectorization and Logistic Regression.

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 91.5% |
| Fake Precision | 91% |
| Real Precision | 92% |
| F1-Score | 0.91 |

## 🛠️ Tech Stack
- Python
- Scikit-learn (TF-IDF + Logistic Regression)
- Pandas & NumPy
- Google Colab

## 📁 Dataset
- **Source:** Fake or Real News Dataset
- **Size:** 6,335 articles
- **Balance:** REAL: 3,171 | FAKE: 3,164 (perfectly balanced)

## 🧠 How It Works
Raw news text
↓
TF-IDF Vectorization (top 5000 words + bigrams)
↓
Logistic Regression Classifier
↓
FAKE ❌ or REAL ✅ + Confidence Score
## ⚠️ Limitations
- Trained on 2016-2017 US political news
- May not generalize well to other domains or time periods
- Dataset bias toward political language

## 🚀 How to Run
1. Open `fake_news_detector.ipynb` in Google Colab
2. Run all cells in order
3. Use `predict_news("your text here")` to test

## 📬 Author
**Tanisha Bharti**  
[LinkedIn](https://linkedin.com/in/tanisha-bharti-80a288269) | 
[Email](mailto:tanibharti343@gmail.com)
