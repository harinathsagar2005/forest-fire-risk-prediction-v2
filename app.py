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
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Load trained models, scalers, and metadata
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics_summary.json')

scaler = joblib.load(SCALER_PATH)
gb_model = joblib.load(os.path.join(MODELS_DIR, 'gb_model.pkl'))
ensemble_model = joblib.load(os.path.join(MODELS_DIR, 'ensemble_model.pkl'))
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
et_model = joblib.load(os.path.join(MODELS_DIR, 'et_model.pkl'))
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
        'project_title': 'Forest Fire Risk Prediction System',
        'models_evaluated': list(metrics_summary['model_comparison'].keys()),
        'best_model': metrics_summary['best_model'],
        'best_accuracy': metrics_summary['best_accuracy'],
        'metrics_summary': metrics_summary,
        'features': FEATURES
    })

@app.route('/api/compare', methods=['GET'])
def get_model_comparison():
    return jsonify({
        'comparison': metrics_summary['model_comparison'],
        'best_model': metrics_summary['best_model'],
        'best_accuracy': metrics_summary['best_accuracy'],
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
        chosen_model = data.get('model', 'Gradient Boosting')

        raw_vector = [temp, rh, rain, wind, press, ndvi, elev, slope, dist_road]
        df_input = pd.DataFrame([raw_vector], columns=FEATURES)
        scaled_vector = scaler.transform(df_input)

        # 1. Gradient Boosting (Rank 1: 99.29%)
        gb_pred = int(gb_model.predict(df_input)[0])
        gb_prob = gb_model.predict_proba(df_input)[0].tolist()

        # 2. Voting Ensemble (Rank 2: 99.14%)
        ens_pred = int(ensemble_model.predict(df_input)[0])
        ens_prob = ensemble_model.predict_proba(df_input)[0].tolist()

        # 3. Random Forest (Rank 3: 97.57%)
        rf_pred = int(rf_model.predict(df_input)[0])
        rf_prob = rf_model.predict_proba(df_input)[0].tolist()

        # 4. Extra Trees (Rank 4: 93.86%)
        et_pred = int(et_model.predict(df_input)[0])
        et_prob = et_model.predict_proba(df_input)[0].tolist()

        # 5. Support Vector Machine (Rank 5: 91.29%)
        svm_pred = int(svm_model.predict(scaled_vector)[0])
        svm_prob = svm_model.predict_proba(scaled_vector)[0].tolist()

        # 6. Logistic Regression (Rank 6: 84.43%)
        lr_pred = int(lr_model.predict(scaled_vector)[0])
        lr_prob = lr_model.predict_proba(scaled_vector)[0].tolist()

        # Selected model resolution
        model_map = {
            'Gradient Boosting': (gb_pred, gb_prob),
            'Voting Ensemble': (ens_pred, ens_prob),
            'Random Forest': (rf_pred, rf_prob),
            'Extra Trees': (et_pred, et_prob),
            'Support Vector Machine': (svm_pred, svm_prob),
            'Logistic Regression': (lr_pred, lr_prob)
        }

        if chosen_model in model_map:
            active_pred, active_prob = model_map[chosen_model]
        else:
            chosen_model = 'Gradient Boosting'
            active_pred, active_prob = gb_pred, gb_prob

        # Continuous risk score percentage (0 to 100)
        continuous_risk_pct = round(
            (active_prob[0] * 12.0 + active_prob[1] * 55.0 + active_prob[2] * 96.0), 1
        )
        continuous_risk_pct = min(100.0, max(0.0, continuous_risk_pct))

        # Local Feature Contribution / SHAP approximation
        contributions = []
        temp_contrib = (temp - 25.0) * 1.8
        contributions.append({'feature': 'Temperature', 'impact': round(temp_contrib, 2), 'unit': '°C', 'val': temp})
        rh_contrib = (48.0 - rh) * 1.6
        contributions.append({'feature': 'Relative Humidity', 'impact': round(rh_contrib, 2), 'unit': '%', 'val': rh})
        rain_contrib = -min(40.0, rain * 8.5)
        contributions.append({'feature': 'Rainfall', 'impact': round(rain_contrib, 2), 'unit': 'mm', 'val': rain})
        wind_contrib = (wind - 14.0) * 1.5
        contributions.append({'feature': 'Wind Speed', 'impact': round(wind_contrib, 2), 'unit': 'km/h', 'val': wind})
        ndvi_contrib = (0.50 - ndvi) * 45.0
        contributions.append({'feature': 'NDVI (Vegetation Index)', 'impact': round(ndvi_contrib, 2), 'unit': 'index', 'val': ndvi})
        slope_contrib = (slope - 10.0) * 1.1
        contributions.append({'feature': 'Slope', 'impact': round(slope_contrib, 2), 'unit': 'deg', 'val': slope})
        road_contrib = max(0.0, (2500.0 - dist_road) / 120.0)
        contributions.append({'feature': 'Distance to Road', 'impact': round(road_contrib, 2), 'unit': 'm', 'val': dist_road})

        contributions = sorted(contributions, key=lambda x: abs(x['impact']), reverse=True)

        comp_dict = metrics_summary.get('model_comparison', {})

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
                'Gradient Boosting': {
                    'prediction': CLASS_LABELS[gb_pred],
                    'prob_high': round(gb_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Gradient Boosting', {}).get('accuracy', 99.29)
                },
                'Voting Ensemble': {
                    'prediction': CLASS_LABELS[ens_pred],
                    'prob_high': round(ens_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Voting Ensemble', {}).get('accuracy', 99.14)
                },
                'Random Forest': {
                    'prediction': CLASS_LABELS[rf_pred],
                    'prob_high': round(rf_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Random Forest', {}).get('accuracy', 97.57)
                },
                'Extra Trees': {
                    'prediction': CLASS_LABELS[et_pred],
                    'prob_high': round(et_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Extra Trees', {}).get('accuracy', 93.86)
                },
                'Support Vector Machine': {
                    'prediction': CLASS_LABELS[svm_pred],
                    'prob_high': round(svm_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Support Vector Machine', {}).get('accuracy', 91.29)
                },
                'Logistic Regression': {
                    'prediction': CLASS_LABELS[lr_pred],
                    'prob_high': round(lr_prob[2] * 100, 1),
                    'accuracy': comp_dict.get('Logistic Regression', {}).get('accuracy', 84.43)
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
