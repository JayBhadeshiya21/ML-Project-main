# CardioGuard - Cardiovascular Risk Prediction Web App

CardioGuard is a Flask-based machine learning web application that estimates cardiovascular disease risk from patient health inputs.

## Overview

The app uses a trained Random Forest model and provides:

- A prediction form for 11 health indicators
- A result modal with risk output
- Dataset explanation pages
- Model workflow and visualization pages

This project is built for educational and demonstration use.

## Tech Stack

- Backend: Python, Flask
- ML: scikit-learn, NumPy, pandas
- Frontend: HTML, CSS, Bootstrap 5, JavaScript (Fetch API)
- Deployment: gunicorn (via `Procfile`)

## Project Structure

```text
.
|-- app.py
|-- model/
|   |-- train_model.py
|   `-- cardio_model.pkl
|-- static/
|   |-- css/
|   |   `-- style.css
|   |-- js/
|   |   `-- script.js
|   `-- plots/
|       |-- confusion_matrix.png
|       |-- feature_importance.png
|       |-- roc_curve.png
|       |-- learning_curve.png
|       |-- age_dist.png
|       `-- prob_dist.png
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- about.html
|   |-- visuals.html
|   |-- model_info.html
|   `-- predict.html
|-- requirements.txt
|-- Procfile
`-- README.md
```

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (if missing)

```bash
python model/train_model.py
```

This creates `model/cardio_model.pkl`.

### 3. Run the app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Routes

- `GET /` -> Home
- `GET /about` -> Dataset details
- `GET /visuals` -> Plots and model visuals
- `GET /model` -> Model workflow
- `GET /predict` -> Prediction page
- `POST /predict` -> JSON prediction endpoint

## Prediction API

### Request

`POST /predict` with JSON body:

```json
{
  "age": 45,
  "gender": 2,
  "height": 175,
  "weight": 70,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

### Response

```json
{
  "prediction": 0,
  "prediction_text": "Normal",
  "probability": 23.4
}
```

## Deployment (Render/Railway style)

- Build command:

```bash
pip install -r requirements.txt && python model/train_model.py
```

- Start command:

```bash
gunicorn app:app
```

`Procfile` already contains the start process.

## Disclaimer

This application is for educational purposes only and is not a substitute for professional medical diagnosis or treatment.





