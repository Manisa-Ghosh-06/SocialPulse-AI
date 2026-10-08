# SocialPulse AI

### An Intelligent Social Media Analytics Framework

SocialPulse AI is an AI-powered social media analytics platform that analyzes uploaded social media data and provides insights into **sentiment, demographics, trending topics, and influence/network connections**.

---
## 📸 Project Preview

![SocialPulse AI Dashboard](dashboard.png)

## 🚀 Features

- 📊 Social media data analysis from CSV files
- 😊 Sentiment analysis using Machine Learning
- 👥 Demographic analysis
- 🔥 Trending topic / narrative analysis
- 🌐 Influence and network analysis
- 📄 Automatic PDF report generation
- 📈 Dashboard-based visualization of analytics

---

## 🧠 Machine Learning

The sentiment analysis module uses:

- **Algorithm:** Logistic Regression
- **Feature Extraction:** Bag of Words (BoW)
- **Text Preprocessing:** Lemmatization
- **N-grams:** Unigram + Bigram
- **Hyperparameter Tuning:** GridSearchCV

The trained sentiment model is used to classify social media posts into:

- Positive
- Negative
- Neutral

---

## 🛠️ Technology Stack

### Backend
- Python
- FastAPI
- Pandas
- Scikit-learn

### Machine Learning / NLP
- NLTK
- Scikit-learn
- Logistic Regression
- Bag of Words

### Report Generation
- ReportLab

### Frontend
- HTML
- CSS
- JavaScript

---

## 🔄 System Workflow

```text
User Uploads CSV
       ↓
CSV Inspection & Column Mapping
       ↓
Data Preprocessing
       ↓
Sentiment Analysis
       ↓
Demographic Analysis
       ↓
Trending Topic Analysis
       ↓
Influence / Network Analysis
       ↓
Dashboard Results
       ↓
PDF Report
