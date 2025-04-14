from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, validators, SelectField, FloatField, IntegerField
from datetime import datetime
import os
import pandas as pd
import numpy as np
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from utils.data_processing import preprocess_data
from utils.visualization import create_visualizations
import config

app = Flask(__name__)
app.config.from_object(config.Config)

# Ensure the database directory exists
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)

# Update the SQLALCHEMY_DATABASE_URI to use absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(db_dir, 'app.db')}"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models (EXACTLY AS YOUR ORIGINAL)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    customer_id = db.Column(db.Integer)
    telecom_company = db.Column(db.String(50))
    region = db.Column(db.String(50))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    contract_type = db.Column(db.String(50))
    contract_duration = db.Column(db.String(50))
    tenure_months = db.Column(db.Integer)
    monthly_charges = db.Column(db.Float)
    data_usage_gb = db.Column(db.Float)
    call_duration_minutes = db.Column(db.Integer)
    complaints_filed = db.Column(db.Integer)
    customer_support_calls = db.Column(db.Integer)
    payment_method = db.Column(db.String(50))
    internet_service = db.Column(db.String(50))
    additional_services = db.Column(db.String(50))
    discount_offer_used = db.Column(db.String(10))
    billing_issues_reported = db.Column(db.Integer)
    prediction = db.Column(db.String(10))
    probability = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Forms (EXACTLY AS YOUR ORIGINAL)
class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email Address', [validators.Length(min=6, max=100)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')

class LoginForm(FlaskForm):
    username = StringField('Username', [validators.DataRequired()])
    password = PasswordField('Password', [validators.DataRequired()])

class PredictionForm(FlaskForm):
    telecom_company = SelectField('Telecom Company', choices=[
        ('Airtel', 'Airtel'), 
        ('Tigo', 'Tigo'), 
        ('Vodacom', 'Vodacom'), 
        ('Halotel', 'Halotel')
    ])
    region = SelectField('Region', choices=[
        ('Dar es Salaam', 'Dar es Salaam'),
        ('Mwanza', 'Mwanza'),
        ('Arusha', 'Arusha'),
        ('Mbeya', 'Mbeya'),
        ('Dodoma', 'Dodoma'),
        ('Tanga', 'Tanga'),
        ('Zanzibar', 'Zanzibar')
    ])
    age = IntegerField('Age', [validators.NumberRange(min=18, max=100)])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')])
    contract_type = SelectField('Contract Type', choices=[
        ('Prepaid', 'Prepaid'), 
        ('Postpaid', 'Postpaid')
    ])
    contract_duration = SelectField('Contract Duration', choices=[
        ('1 Month', '1 Month'),
        ('6 Months', '6 Months'),
        ('12 Months', '12 Months'),
        ('24 Months', '24 Months')
    ])
    tenure_months = IntegerField('Tenure (Months)', [validators.NumberRange(min=1, max=120)])
    monthly_charges = FloatField('Monthly Charges', [validators.NumberRange(min=0)])
    data_usage_gb = FloatField('Data Usage (GB)', [validators.NumberRange(min=0)])
    call_duration_minutes = IntegerField('Call Duration (Minutes)', [validators.NumberRange(min=0)])
    complaints_filed = IntegerField('Complaints Filed', [validators.NumberRange(min=0)])
    customer_support_calls = IntegerField('Customer Support Calls', [validators.NumberRange(min=0)])
    payment_method = SelectField('Payment Method', choices=[
        ('Credit Card', 'Credit Card'),
        ('Bank Transfer', 'Bank Transfer'),
        ('Mobile Money', 'Mobile Money'),
        ('Cash', 'Cash')
    ])
    internet_service = SelectField('Internet Service', choices=[
        ('Mobile Data', 'Mobile Data'),
        ('Fiber', 'Fiber'),
        ('DSL', 'DSL'),
        ('None', 'None')
    ])
    additional_services = SelectField('Additional Services', choices=[
        ('Streaming', 'Streaming'),
        ('VPN', 'VPN'),
        ('Cloud Storage', 'Cloud Storage'),
        ('None', 'None')
    ])
    discount_offer_used = SelectField('Discount Offer Used', choices=[
        ('Yes', 'Yes'),
        ('No', 'No')
    ])
    billing_issues_reported = IntegerField('Billing Issues Reported', [validators.NumberRange(min=0)])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the trained model (EXACTLY AS YOUR ORIGINAL)
try:
    model = joblib.load('models/churn_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please ensure 'models/churn_model.pkl' exists.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Routes (ONLY UPDATED REGISTRATION/LOGIN TO FIX ISSUES)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            # Only added duplicate check - rest is original
            if User.query.filter_by(username=form.username.data).first():
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
                
            if User.query.filter_by(email=form.email.data).first():
                flash('Email already exists', 'danger')
                return redirect(url_for('register'))

            # YOUR ORIGINAL REGISTRATION CODE
            hashed_password = generate_password_hash(form.password.data, method='sha256')
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password_hash=hashed_password
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {e}', 'danger')  # Added error details
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        # Only added more specific error messages - logic is original
        if not user:
            flash('Username not found', 'danger')
            return redirect(url_for('login'))
            
        if not check_password_hash(user.password_hash, form.password.data):
            flash('Incorrect password', 'danger')
            return redirect(url_for('login'))

        # YOUR ORIGINAL LOGIN CODE
        login_user(user)
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('index'))
    return render_template('login.html', form=form)

# ALL OTHER ROUTES REMAIN EXACTLY AS YOUR ORIGINAL
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        try:
            input_data = {
                'TelecomCompany': form.telecom_company.data,
                'Region': form.region.data,
                'Age': form.age.data,
                'Gender': form.gender.data,
                'ContractType': form.contract_type.data,
                'ContractDuration': form.contract_duration.data,
                'TenureMonths': form.tenure_months.data,
                'MonthlyCharges': form.monthly_charges.data,
                'DataUsageGB': form.data_usage_gb.data,
                'CallDurationMinutes': form.call_duration_minutes.data,
                'ComplaintsFiled': form.complaints_filed.data,
                'CustomerSupportCalls': form.customer_support_calls.data,
                'PaymentMethod': form.payment_method.data,
                'InternetService': form.internet_service.data,
                'AdditionalServices': form.additional_services.data,
                'DiscountOfferUsed': form.discount_offer_used.data,
                'BillingIssuesReported': form.billing_issues_reported.data
            }
            
            input_df = pd.DataFrame([input_data])
            processed_data = preprocess_data(input_df)
            
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            new_prediction = Prediction(
                user_id=current_user.id,
                telecom_company=form.telecom_company.data,
                region=form.region.data,
                age=form.age.data,
                gender=form.gender.data,
                contract_type=form.contract_type.data,
                contract_duration=form.contract_duration.data,
                tenure_months=form.tenure_months.data,
                monthly_charges=form.monthly_charges.data,
                data_usage_gb=form.data_usage_gb.data,
                call_duration_minutes=form.call_duration_minutes.data,
                complaints_filed=form.complaints_filed.data,
                customer_support_calls=form.customer_support_calls.data,
                payment_method=form.payment_method.data,
                internet_service=form.internet_service.data,
                additional_services=form.additional_services.data,
                discount_offer_used=form.discount_offer_used.data,
                billing_issues_reported=form.billing_issues_reported.data,
                prediction='Yes' if prediction == 1 else 'No',
                probability=probability
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            
            result = {
                'prediction': 'Yes' if prediction == 1 else 'No',
                'probability': f"{probability * 100:.2f}%",
                'recommendation': 'High risk of churn. Consider retention strategies.' if prediction == 1 else 'Low risk of churn.'
            }
            
            return render_template('predict.html', form=form, result=result, prediction_id=new_prediction.id)
            
        except Exception as e:
            db.session.rollback()
            flash('Prediction failed. Please try again.', 'danger')
            app.logger.error(f"Prediction error: {str(e)}")
    
    return render_template('predict.html', form=form)

@app.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('predictions.html', predictions=user_predictions)

@app.route('/analysis')
@login_required
def analysis():
    viz_data = create_visualizations()
    return render_template('analysis.html', viz_data=viz_data)

@app.route('/export_predictions/<format>')
@login_required
def export_predictions(format):
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        
        if not predictions:
            flash('No predictions found to export', 'warning')
            return redirect(url_for('predictions'))
        
        data = []
        for pred in predictions:
            data.append({
                'Timestamp': pred.timestamp,
                'Telecom Company': pred.telecom_company,
                'Region': pred.region,
                'Age': pred.age,
                'Gender': pred.gender,
                'Contract Type': pred.contract_type,
                'Contract Duration': pred.contract_duration,
                'Tenure (Months)': pred.tenure_months,
                'Monthly Charges': pred.monthly_charges,
                'Data Usage (GB)': pred.data_usage_gb,
                'Call Duration (Minutes)': pred.call_duration_minutes,
                'Complaints Filed': pred.complaints_filed,
                'Customer Support Calls': pred.customer_support_calls,
                'Payment Method': pred.payment_method,
                'Internet Service': pred.internet_service,
                'Additional Services': pred.additional_services,
                'Discount Offer Used': pred.discount_offer_used,
                'Billing Issues Reported': pred.billing_issues_reported,
                'Prediction': pred.prediction,
                'Probability': pred.probability
            })
        
        df = pd.DataFrame(data)
        
        if format == 'csv':
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='churn_predictions.csv'
            )
        elif format == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='churn_predictions.xlsx'
            )
        else:
            flash('Invalid export format', 'danger')
            return redirect(url_for('predictions'))
            
    except Exception as e:
        flash('Export failed. Please try again.', 'danger')
        app.logger.error(f"Export error: {str(e)}")
        return redirect(url_for('predictions'))

@app.route('/download_pdf/<int:prediction_id>')
@login_required
def download_pdf(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.drawString(100, 750, "Tanzania Telecom Churn Prediction Report")
    p.drawString(100, 730, f"Prediction ID: {prediction.id}")
    p.drawString(100, 710, f"Date: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(100, 690, "Customer Details:")
    p.drawString(120, 670, f"Telecom Company: {prediction.telecom_company}")
    p.drawString(120, 650, f"Region: {prediction.region}")
    p.drawString(120, 630, f"Age: {prediction.age}")
    p.drawString(120, 610, f"Gender: {prediction.gender}")
    p.drawString(120, 590, f"Contract Type: {prediction.contract_type}")
    p.drawString(120, 570, f"Contract Duration: {prediction.contract_duration}")
    p.drawString(120, 550, f"Tenure (Months): {prediction.tenure_months}")
    p.drawString(120, 530, f"Monthly Charges: {prediction.monthly_charges}")
    p.drawString(120, 510, f"Data Usage (GB): {prediction.data_usage_gb}")
    p.drawString(120, 490, f"Call Duration (Minutes): {prediction.call_duration_minutes}")
    p.drawString(120, 470, f"Complaints Filed: {prediction.complaints_filed}")
    p.drawString(120, 450, f"Customer Support Calls: {prediction.customer_support_calls}")
    p.drawString(120, 430, f"Payment Method: {prediction.payment_method}")
    p.drawString(120, 410, f"Internet Service: {prediction.internet_service}")
    p.drawString(120, 390, f"Additional Services: {prediction.additional_services}")
    p.drawString(120, 370, f"Discount Offer Used: {prediction.discount_offer_used}")
    p.drawString(120, 350, f"Billing Issues Reported: {prediction.billing_issues_reported}")
    p.drawString(100, 330, "Prediction Results:")
    p.drawString(120, 310, f"Churn Prediction: {prediction.prediction}")
    p.drawString(120, 290, f"Probability: {prediction.probability * 100:.2f}%")
    p.drawString(100, 270, "Recommendation:")
    
    if prediction.prediction == 'Yes':
        p.drawString(120, 250, "High risk of churn. Consider retention strategies:")
        p.drawString(140, 230, "- Offer personalized discounts or promotions")
        p.drawString(140, 210, "- Improve customer service interactions")
        p.drawString(140, 190, "- Address billing issues promptly")
        p.drawString(140, 170, "- Provide value-added services")
    else:
        p.drawString(120, 250, "Low risk of churn. Maintain current service quality.")
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'churn_prediction_{prediction.id}.pdf'
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        required_fields = [
            'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType', 
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        input_df = pd.DataFrame([data])
        processed_data = preprocess_data(input_df)
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully")
        except Exception as e:
            print(f"Error creating database: {str(e)}")
            try:
                open(os.path.join(db_dir, 'app.db'), 'w').close()
                db.create_all()
                print("Database created after manual intervention")
            except Exception as e2:
                print(f"Failed to create database: {str(e2)}")
                raise
    
    app.run(debug=True)