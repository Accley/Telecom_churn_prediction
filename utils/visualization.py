import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
from utils.model_training import get_feature_importance

def create_visualizations():
    # Load the data
    df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
    
    # Create visualizations
    viz_data = {}
    
    # Churn distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    churn_dist = plot_to_base64(plt)
    plt.close()
    viz_data['churn_dist'] = churn_dist
    
    # Churn by Telecom Company
    plt.figure(figsize=(10, 6))
    sns.countplot(x='TelecomCompany', hue='Churn', data=df)
    plt.title('Churn by Telecom Company')
    churn_by_company = plot_to_base64(plt)
    plt.close()
    viz_data['churn_by_company'] = churn_by_company
    
    # Churn by Contract Type
    plt.figure(figsize=(8, 6))
    sns.countplot(x='ContractType', hue='Churn', data=df)
    plt.title('Churn by Contract Type')
    churn_by_contract = plot_to_base64(plt)
    plt.close()
    viz_data['churn_by_contract'] = churn_by_contract
    
    # Age distribution by Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='Age', data=df)
    plt.title('Age Distribution by Churn')
    age_dist = plot_to_base64(plt)
    plt.close()
    viz_data['age_dist'] = age_dist
    
    # Tenure distribution by Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='TenureMonths', data=df)
    plt.title('Tenure Distribution by Churn')
    tenure_dist = plot_to_base64(plt)
    plt.close()
    viz_data['tenure_dist'] = tenure_dist
    
    # Monthly Charges by Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('Monthly Charges by Churn')
    charges_dist = plot_to_base64(plt)
    plt.close()
    viz_data['charges_dist'] = charges_dist
    
    # Feature importance
    feature_importance = get_feature_importance()
    if feature_importance is not None:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features for Churn Prediction')
        feature_imp = plot_to_base64(plt)
        plt.close()
        viz_data['feature_imp'] = feature_imp
    
    return viz_data

def plot_to_base64(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64