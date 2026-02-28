import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np

def load_data():
    try:
        # Try loading from parent directory (standard structure)
        df = pd.read_csv('ds_cvd_w1.csv')
    except Exception:
        try:
            # Fallback for different CWD
            df = pd.read_csv('../ds_cvd_w1.csv')
        except:
            return None
    
    # Preprocessing for visuals
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Convert age to years if needed
    if df['age'].mean() > 150:
        df['age'] = (df['age'] / 365.25).round().astype(int)
        
    return df

def get_base64_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_target_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='cardio', data=df, palette='coolwarm')
    plt.title('Distribution of Heart Disease (Target)')
    plt.xlabel('Condition (0: Healthy, 1: Risk)')
    plt.ylabel('Count')
    return get_base64_plot()

def plot_age_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age (Years)')
    return get_base64_plot()

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    return get_base64_plot()

def plot_lifestyle_risk(df):
    # Analyzing Smoke, Alcohol, Active vs Cardio
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.barplot(x='smoke', y='cardio', data=df, ax=axes[0], palette='Blues')
    axes[0].set_title('Smoking vs Risk')
    
    sns.barplot(x='alco', y='cardio', data=df, ax=axes[1], palette='Greens')
    axes[1].set_title('Alcohol vs Risk')
    
    sns.barplot(x='active', y='cardio', data=df, ax=axes[2], palette='Reds')
    axes[2].set_title('Activity vs Risk')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')
