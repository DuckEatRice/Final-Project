# 🧠 Teen Stress Level Predictor

A machine learning web app that predicts a teenager's stress level (1–10) based on their daily habits and lifestyle.

---

## 👥 Team
- Joshua Huang
- Ethan Chang

---

## 📌 Overview

Teenagers are increasingly affected by social media and screen time. This project uses behavioral data — sleep, social media usage, academic performance, and more — to predict a teen's stress level using regression models.

The final app is deployed on **Hugging Face Spaces** using **Gradio**.

---

## 📊 Dataset

- **Source:** [Teenager Mental Health Dataset](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) (Kaggle)
- **Contains:** Teen behavioral and mental health data including social media habits, sleep, academics, and depression indicators

---

## 🔢 Features Used

| Feature | Description |
|---|---|
| Age | 13–19 |
| Gender | Male / Female / Other |
| Daily Social Media Hours | Hours per day |
| Platform Usage | e.g. Instagram, TikTok, Both |
| Sleep Hours | Hours per night |
| Screen Time Before Sleep | Hours before bed |
| Academic Performance | GPA (2.0–4.0) |
| Physical Activity | Hours per day |
| Social Interaction Level | High / Medium / Low |
| Depression Label | 0 = No, 1 = Yes |

**Target:** `stress_level` — engineered score from 1 to 10

---

## 🤖 Models Trained

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected automatically by R² score.

---

## 📈 Results

| Metric | Value |
|---|---|
| Best Model | Linear Regression |
| R² | 0.963 |
| MAE | 0.260 |

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/teen-stress-predictor.git
cd teen-stress-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Make sure `Teen_Mental_Health_Dataset.csv` is in the root directory.

---

## 📦 Requirements

```
gradio
pandas
numpy
scikit-learn
plotly
```

---

## ⚠️ Disclaimer

This app is for **educational purposes only** and should not be used as a clinical or diagnostic tool.
