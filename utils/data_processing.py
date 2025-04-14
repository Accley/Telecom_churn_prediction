import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(df):
    # Load the preprocessing objects
    label_encoders = joblib.load('models/label_encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Apply label encoding to categorical features
    categorical_cols = [
        'TelecomCompany', 'Region', 'Gender', 'ContractType', 
        'ContractDuration', 'PaymentMethod', 'InternetService', 
        'AdditionalServices', 'DiscountOfferUsed'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])
    
    # Scale numerical features
    numerical_cols = [
        'Age', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
        'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
        'BillingIssuesReported'
    ]
    
    if numerical_cols[0] in df.columns:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def prepare_training_data():
    # Load the dataset
    df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
    
    # Drop CustomerID as it's not a feature
    df = df.drop('CustomerID', axis=1)
    
    # Convert Churn to binary (1 for 'Yes', 0 for 'No')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical and numerical columns
    categorical_cols = [
        'TelecomCompany', 'Region', 'Gender', 'ContractType', 
        'ContractDuration', 'PaymentMethod', 'InternetService', 
        'AdditionalServices', 'DiscountOfferUsed'
    ]
    
    numerical_cols = [
        'Age', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
        'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
        'BillingIssuesReported'
    ]
    
    # Label encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Save the label encoders for later use
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test