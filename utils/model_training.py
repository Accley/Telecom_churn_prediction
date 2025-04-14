from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import joblib
from utils.data_processing import prepare_training_data

def train_and_evaluate_models():
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_training_data()
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate models
    results = []
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'CV Mean Accuracy': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Test Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Save the best model (XGBoost based on initial evaluation)
        if name == 'XGBoost':
            joblib.dump(model, 'models/churn_model.pkl')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def get_feature_importance():
    # Load the best model
    model = joblib.load('models/churn_model.pkl')
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = model.coef_[0]
    else:
        return None
    
    # Get feature names
    df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
    features = df.drop(['CustomerID', 'Churn'], axis=1).columns
    
    # Create DataFrame of feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return feature_importance