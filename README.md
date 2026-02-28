# CardioGuard - Cardiovascular Disease Prediction

## 🏥 Overview
CardioGuard is a production-ready Web Application that predicts the likelihood of cardiovascular disease using Machine Learning (Random Forest). It features a modern, responsive UI designed with a healthcare theme and provides instant risk analysis based on user health metrics.

## 🚀 Features
- **Machine Learning Backend**: Built with Flask and Scikit-Learn.
- **Modern UI**: Responsive layout with branded colours, gradient buttons, and color‑mode toggle (light/dark).
- **Interactive Form**: Easy-to-use input for 11 health indicators.
- **Immediate Results**: Real-time prediction with clear risk visualization.
- **Rich Content Pages**: Dataset details, visualizations and workflow explained with cards and tabs.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3 (custom variables & dark-mode), Bootstrap 5, JavaScript (Fetch API)
- **Backend**: Python (Flask)
- **ML**: Scikit-Learn (Random Forest Classifier), Pandas, NumPy
- **Deployment**: Render / Railway (gunicorn)

## 📂 Project Structure

*Note: design updates include a white/light theme by default with an optional dark overlay that can be toggled via the moon/sun button in the navbar.*
```
/
├── app.py                # Flask Application
├── model/
│   ├── train_model.py    # Script to train and save the model
│   └── cardio_model.pkl  # Trained ML Model (Generated)
├── static/
│   ├── css/style.css     # Custom Styles
│   └── js/script.js      # Frontend Logic
├── templates/
│   ├── base.html          # Base layout used across all pages
│   ├── home.html          # Website landing page (extends base)
│   └── ... other pages
├── requirements.txt      # Dependencies
├── Procfile              # Deployment Command
└── README.md             # Documentation
```

## ⚙️ Setup & Installation

### Prerequisite
Ensure you have Python 3.8+ installed.

1. **Clone or Download the Project**
   ```bash
   git clone <repo_url>
   cd <project_folder>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model** (Important!)
   You must generate the model file first.
   ```bash
   python model/train_model.py
   ```
   *This will create `model/cardio_model.pkl`.*

4. **Run the Application**
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`.

## ☁️ Deployment Guide (Render)

1. **Push to GitHub**: Upload this code to a GitHub repository.
2. **Create New Web Service**: Go to [Render Dashboard](https://dashboard.render.com/) -> New -> Web Service.
3. **Connect Repo**: Select your repository.
4. **Settings**:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python model/train_model.py` (Adding training here ensures model exists on cloud)
   - **Start Command**: `gunicorn app:app`
5. **Deploy**: Click Create Web Service.

## 📸 Screenshots
*(Add screenshots here after running the app)*

## ⚠️ Disclaimer
This application is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
