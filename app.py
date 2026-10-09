"""
Forest Fire Risk Prediction Using Machine Learning
Full-Stack Web Application Backend (Flask)
Vardhaman College of Engineering (Autonomous)
Department of Computer Science and Engineering (AI&ML)
Team Number: 27
Guide: Mrs. D Sandhya Rani, Assistant Professor
Students: G. Harinath (24881A6619), S. Anirudh (24881A6649)
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from gis_engine import get_gis_forest_zones
from llm_advisor import generate_llm_incident_report, answer_wildfire_copilot_chat

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load trained models, scalers, and metadata
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics_summary.json')

scaler = joblib.load(SCALER_PATH)
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
lr_model = joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl'))

with open(METRICS_PATH, 'r') as f:
    metrics_summary = json.load(f)

FEATURES = metrics_summary.get('features', [
    'temperature', 'relative_humidity', 'rainfall',
    'wind_speed', 'atmospheric_pressure', 'ndvi',
    'elevation', 'slope', 'distance_to_road'
])

CLASS_LABELS = {
    0: 'Low Risk',
    1: 'Medium Risk',
    2: 'High Risk'
}

CLASS_COLORS = {
    0: '#10b981', # Green
    1: '#f59e0b', # Amber
    2: '#ef4444'  # Red
}

ACTION_RECOMMENDATIONS = {
    0: {
        'status': 'Normal / Safe Status',
        'alert_level': 'Green Condition',
        'actions': [
            'Maintain standard environmental monitoring and routine telemetry check.',
            'Keep forest patrol on standard operating schedule.',
            'No fuel containment or water tender mobilization required.'
        ]
    },
    1: {
        'status': 'Caution / Elevated Susceptibility',
        'alert_level': 'Yellow Alert',
        'actions': [
            'Activate forest watch towers and drone surveillance over buffer zones.',
            'Inspect fire break lines and maintain high readiness at nearby stations.',
            'Issue advisory warning to surrounding agricultural and roadside communities.'
        ]
    },
    2: {
        'status': 'Critical Wildfire Hazard',
        'alert_level': 'Red Emergency',
        'actions': [
            'Trigger automated siren alarms and notify District Forest Disaster Response Team.',
            'Deploy rapid-intervention water tankers and pre-position counter-fire teams.',
            'Restrict public access to nearby trails, roads, and forested tourist corridors.',
            'Continuous satellite thermal tracking and real-time wind spread vector analysis.'
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report')
def outcome_report():
    return render_template('report.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        'college': 'Vardhaman College of Engineering (Autonomous)',
        'accreditation': 'Affiliated to JNTUH, Approved by AICTE, Accredited by NAAC A++, ISO 9001:2015',
        'location': 'Kacharam, Shamshabad, Hyderabad – 501218, Telangana, India',
        'department': 'Department of Computer Science and Engineering (AI&ML)',
        'academic_year': 'A. Y. 2026-2027',
        'project_title': 'Forest Fire Risk Prediction Using Machine Learning',
        'team_number': 27,
        'team_members': [
            {'name': 'G. Harinath', 'roll_no': '24881A6619'},
            {'name': 'S. Anirudh', 'roll_no': '24881A6649'}
        ],
        'guide': {
            'name': 'Mrs. D Sandhya Rani',
            'designation': 'Assistant Professor'
        },
        'objectives': [
            'To Develop a machine learning model to predict forest fire risk using historical fire and environmental data.',
            'To Compare Logistic Regression, Random Forest, and Support Vector Machine models to identify the best-performing algorithm.',
            'To Generate forest fire risk maps to support early warning, resource allocation, and disaster prevention.'
        ],
        'metrics_summary': metrics_summary,
        'features': FEATURES
    })

@app.route('/api/compare', methods=['GET'])
def get_model_comparison():
    return jsonify({
        'comparison': metrics_summary['model_comparison'],
        'best_model': metrics_summary['best_model'],
        'feature_importance': metrics_summary['feature_importance'],
        'shap_importance': metrics_summary['shap_importance']
    })

@app.route('/api/gis-data', methods=['GET'])
def get_gis_data():
    zones = get_gis_forest_zones()
    return jsonify({'zones': zones})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Extract features with sensible defaults
        temp = float(data.get('temperature', 28.0))
        rh = float(data.get('relative_humidity', 45.0))
        rain = float(data.get('rainfall', 0.0))
        wind = float(data.get('wind_speed', 15.0))
        press = float(data.get('atmospheric_pressure', 1012.0))
        ndvi = float(data.get('ndvi', 0.45))
        elev = float(data.get('elevation', 500.0))
        slope = float(data.get('slope', 12.0))
        dist_road = float(data.get('distance_to_road', 1500.0))
        chosen_model = data.get('model', 'Random Forest')

        raw_vector = [temp, rh, rain, wind, press, ndvi, elev, slope, dist_road]
        df_input = pd.DataFrame([raw_vector], columns=FEATURES)
        scaled_vector = scaler.transform(df_input)

        # Predict with all 3 models for comparative real-time evaluation
        # Random Forest
        rf_pred = int(rf_model.predict(df_input)[0])
        rf_prob = rf_model.predict_proba(df_input)[0].tolist()

        # Support Vector Machine
        svm_pred = int(svm_model.predict(scaled_vector)[0])
        svm_prob = svm_model.predict_proba(scaled_vector)[0].tolist()

        # Logistic Regression
        lr_pred = int(lr_model.predict(scaled_vector)[0])
        lr_prob = lr_model.predict_proba(scaled_vector)[0].tolist()

        # Selected model resolution
        if chosen_model == 'Support Vector Machine':
            active_pred = svm_pred
            active_prob = svm_prob
        elif chosen_model == 'Logistic Regression':
            active_pred = lr_pred
            active_prob = lr_prob
        else:
            chosen_model = 'Random Forest'
            active_pred = rf_pred
            active_prob = rf_prob

        # Continuous risk score percentage (0 to 100)
        # Low risk weight 10%, Medium 50%, High 95%
        continuous_risk_pct = round(
            (active_prob[0] * 12.0 + active_prob[1] * 55.0 + active_prob[2] * 96.0), 1
        )
        continuous_risk_pct = min(100.0, max(0.0, continuous_risk_pct))

        # Local Feature Contribution / SHAP approximation
        # Show how each feature pushed the risk upward or downward
        contributions = []
        # Temperature baseline ~ 25C
        temp_contrib = (temp - 25.0) * 1.8
        contributions.append({'feature': 'Temperature', 'impact': round(temp_contrib, 2), 'unit': '°C', 'val': temp})
        # Humidity baseline ~ 50% (lower is dangerous)
        rh_contrib = (48.0 - rh) * 1.6
        contributions.append({'feature': 'Relative Humidity', 'impact': round(rh_contrib, 2), 'unit': '%', 'val': rh})
        # Rainfall baseline ~ 2mm (rain heavily suppresses)
        rain_contrib = -min(40.0, rain * 8.5)
        contributions.append({'feature': 'Rainfall', 'impact': round(rain_contrib, 2), 'unit': 'mm', 'val': rain})
        # Wind speed baseline ~ 12 km/h
        wind_contrib = (wind - 14.0) * 1.5
        contributions.append({'feature': 'Wind Speed', 'impact': round(wind_contrib, 2), 'unit': 'km/h', 'val': wind})
        # NDVI baseline ~ 0.5 (lower dry fuel increases fire risk)
        ndvi_contrib = (0.50 - ndvi) * 45.0
        contributions.append({'feature': 'NDVI (Vegetation Index)', 'impact': round(ndvi_contrib, 2), 'unit': 'index', 'val': ndvi})
        # Slope baseline ~ 10 deg (steeper slope creates upward chimney effect)
        slope_contrib = (slope - 10.0) * 1.1
        contributions.append({'feature': 'Slope', 'impact': round(slope_contrib, 2), 'unit': 'deg', 'val': slope})
        # Road proximity (closer is more prone to ignition)
        road_contrib = max(0.0, (2500.0 - dist_road) / 120.0)
        contributions.append({'feature': 'Distance to Road', 'impact': round(road_contrib, 2), 'unit': 'm', 'val': dist_road})

        contributions = sorted(contributions, key=lambda x: abs(x['impact']), reverse=True)

        return jsonify({
            'status': 'success',
            'chosen_model': chosen_model,
            'prediction': {
                'class_index': active_pred,
                'class_name': CLASS_LABELS[active_pred],
                'risk_percentage': continuous_risk_pct,
                'color': CLASS_COLORS[active_pred],
                'probabilities': {
                    'Low Risk': round(active_prob[0] * 100, 1),
                    'Medium Risk': round(active_prob[1] * 100, 1),
                    'High Risk': round(active_prob[2] * 100, 1)
                }
            },
            'all_models': {
                'Random Forest': {
                    'prediction': CLASS_LABELS[rf_pred],
                    'prob_high': round(rf_prob[2] * 100, 1),
                    'accuracy': metrics_summary['model_comparison']['Random Forest']['accuracy']
                },
                'Support Vector Machine': {
                    'prediction': CLASS_LABELS[svm_pred],
                    'prob_high': round(svm_prob[2] * 100, 1),
                    'accuracy': metrics_summary['model_comparison']['Support Vector Machine']['accuracy']
                },
                'Logistic Regression': {
                    'prediction': CLASS_LABELS[lr_pred],
                    'prob_high': round(lr_prob[2] * 100, 1),
                    'accuracy': metrics_summary['model_comparison']['Logistic Regression']['accuracy']
                }
            },
            'feature_contributions': contributions,
            'disaster_management': ACTION_RECOMMENDATIONS[active_pred]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/historical-sample', methods=['GET'])
def get_historical_sample():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'forest_fire_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        sample = df.sample(n=min(25, len(df)), random_state=42).to_dict(orient='records')
        return jsonify({'total_records': len(df), 'sample': sample})
    return jsonify({'total_records': 0, 'sample': []})

@app.route('/api/llm/incident-report', methods=['POST'])
def get_llm_incident_report():
    try:
        data = request.get_json(force=True)
        api_key = data.get('api_key', None)
        result = generate_llm_incident_report(data, api_key=api_key)
        return jsonify({
            'status': 'success',
            'source': result['source'],
            'report_markdown': result['report_text']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/llm/chat', methods=['POST'])
def handle_llm_chat():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400
        context = data.get('context', {})
        api_key = data.get('api_key', None)
        result = answer_wildfire_copilot_chat(message, current_context=context, api_key=api_key)
        return jsonify({
            'status': 'success',
            'source': result['source'],
            'reply': result['answer']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print(f"[*] Starting Forest Fire Risk Prediction Web Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
