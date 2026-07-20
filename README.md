# 🎬 GenreLens — AI Movie Genre Predictor

A multi-label movie genre classifier that predicts genres from movie poster images using deep learning.

**Live Demo**: [GenreLens](https://moviegenre-predictor.netlify.app)  
**Demo Video**: [Drive Link](https://drive.google.com/file/d/13U9ErnCTnU3GpUpx_gfMzFvnpUfPzRJU/view?usp=sharing) 
**Backend API**: [Render](https://movie-genre-predictor-mqbb.onrender.com)

---

## 🚀 Features

- Upload any movie poster and get predicted genres with confidence scores
- Multi-label classification — predicts multiple genres simultaneously
- Built with EfficientNetB0 transfer learning
- REST API backend with Django REST Framework
- Clean cinema-themed frontend with drag and drop support

---

## 🧠 ML Pipeline

| Step | Details |
|------|---------|
| Dataset | 3,792 movie posters from Kaggle (neha1703/movie-genre-from-its-poster) |
| Genres | 8 genres — Action, Comedy, Crime, Documentary, Drama, Horror, Romance, Thriller |
| Base Model | EfficientNetB0 (ImageNet pretrained, frozen) |
| Output Layer | Dense(8, activation='sigmoid') |
| Loss Function | Binary Cross-Entropy |
| Evaluation Metric | Macro F1 Score |
| Final Macro F1 | 0.30 |

---

## 📊 Experiments

| Setup | Macro F1 |
|-------|----------|
| MobileNetV2 frozen, 22 genres | 0.07 |
| MobileNetV2 frozen, 11 genres | 0.15 |
| MobileNetV2 frozen, 8 genres | 0.23 |
| MobileNetV2 fine-tuned, 8 genres | 0.19 |
| EfficientNetB0 frozen, 8 genres | **0.30** ← best |

---

## 🏗️ Architecture

---

## 🔧 Tech Stack

- **ML**: TensorFlow, Keras, EfficientNetB0, scikit-learn
- **Backend**: Django, Django REST Framework, Gunicorn
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **Deployment**: Render (backend), Netlify (frontend)
- **Data**: Pandas, NumPy, Matplotlib, Seaborn

---

## 📁 Project Structure

---

## 🚀 Run Locally

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

**Frontend:**
```bash
# Open frontend/index.html in browser
# or use Live Server extension in VS Code
```

---

## 🔍 How It Works

The model analyzes visual patterns in movie posters:
- 🌑 Dark tones → Horror / Thriller
- 🎭 Bright colors → Comedy
- 💥 Action poses → Action
- 💑 Couple close-up → Romance

> **Note**: Model reads pixel patterns only — it cannot read text on posters. This is a known limitation. Planned improvement: OCR + IMDB API for hybrid classification.

---

## 📈 Key Decisions

- **Multi-label over single-label**: Movies belong to multiple genres simultaneously (avg 2.18 genres per poster)
- **Sigmoid over softmax**: Each genre is an independent yes/no decision
- **Macro F1 over accuracy**: Dataset is imbalanced — Drama (1,963) vs Documentary (385)
- **8 genres only**: Genres below 30 examples are unlearnable — removed Talk-Show, Game-Show etc.
- **Frozen base model**: Only 3,792 images — fine-tuning caused overfitting (F1 dropped to 0.19)

---

## 🙋 Author

**Ardra** — CS Engineering Student, College of Engineering Thalassery  
[GitHub](https://github.com/ardra-p)