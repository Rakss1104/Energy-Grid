import os
import json
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import uuid

# Import your MCP tools (assuming they're in updated_mcp_server.py)
from mcp_Server import (
    predict_consumption, prep_energy_3h, monitor_generation, optimize_load,
    analyze_consumption_patterns, get_similar_consumption_days,
    ForecastRequest, TimePoint, Prep3hRequest, LiveSource, MonitorRequest, OptimizeRequest
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')

@app.route('/')
def index():
    """Main dashboard with all tools"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for consumption prediction"""
    try:
        data = request.json

        # Create request object with new schema
        req = ForecastRequest(
            city=data['city'],
            area=data.get('area'),
            horizon_hours=int(data.get('horizon_hours', 24)),
            weather_hint=data.get('weather_hint')
        )

        # Call the prediction tool
        result = predict_consumption(req)

        # Convert plot to base64 for web display
        with open(result['plot_path'], 'rb') as img_file:
            plot_data = base64.b64encode(img_file.read()).decode()

        # Clean up the temporary file
        os.remove(result['plot_path'])

        return jsonify({
            'success': True,
            'data': {
                **result,
                'plot_base64': plot_data
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/prep', methods=['POST'])
def api_prep():
    """API endpoint for 3-hour energy preparation"""
    try:
        data = request.json

        req = Prep3hRequest(
            city=data['city'],
            area=data.get('area'),
            current_load_mw=float(data['current_load_mw']),
            reserve_margin_pct=float(data.get('reserve_margin_pct', 15.0)),
            notes=data.get('notes')
        )

        result = prep_energy_3h(req)

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/monitor', methods=['POST'])
def api_monitor():
    """API endpoint for generation monitoring"""
    try:
        data = request.json

        # Convert sources to LiveSource objects
        sources = []
        for source_data in data['sources']:
            sources.append(LiveSource(
                name=source_data['name'],
                kind=source_data['kind'],
                current_mw=float(source_data['current_mw']),
                expected_mw=float(source_data.get('expected_mw', 0)) or None
            ))

        req = MonitorRequest(
            window_minutes=int(data.get('window_minutes', 15)),
            sources=sources
        )

        result = monitor_generation(req)

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API endpoint for load optimization"""
    try:
        data = request.json

        req = OptimizeRequest(
            city=data['city'],
            area=data.get('area'),
            demand_forecast_mw=[float(x) for x in data['demand_forecast_mw']],
            dispatchable_mw=float(data['dispatchable_mw']),
            renewables_now_mw=float(data['renewables_now_mw']),
            constraints=data.get('constraints')
        )

        result = optimize_load(req)

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analyze-patterns', methods=['POST'])
def api_analyze_patterns():
    """API endpoint for consumption pattern analysis"""
    try:
        data = request.json

        result = analyze_consumption_patterns(
            city=data['city'],
            area=data.get('area')
        )

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/similar-days', methods=['POST'])
def api_similar_days():
    """API endpoint for finding similar consumption days"""
    try:
        data = request.json

        result = get_similar_consumption_days(
            city=data['city'],
            target_date=data['target_date'],
            area=data.get('area'),
            n_days=int(data.get('n_days', 5))
        )

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/sample-data')
def sample_data():
    """Generate sample data for testing based on new dataset format"""

    # Cities and areas based on the new dataset
    cities = ["Bangalore"]
    areas = ["MG Road", "Brigade Road", "Church Street", "Richmond Town",
             "Shivajinagar", "Vasanth Nagar", "Ulsoor (Halasuru)",
             "Indiranagar", "Koramangala", "Malleshwaram", "Rajajinagar"]

    # Sample renewable sources
    sources = [
        {'name': 'Solar Farm Alpha', 'kind': 'solar', 'current_mw': 180, 'expected_mw': 220},
        {'name': 'Wind Farm Beta', 'kind': 'wind', 'current_mw': 85, 'expected_mw': 120},
        {'name': 'Rooftop Solar Network', 'kind': 'rooftop', 'current_mw': 45, 'expected_mw': 50},
        {'name': 'Hydro Plant Gamma', 'kind': 'hydro', 'current_mw': 150, 'expected_mw': 150},
    ]

    # Sample forecast data (6 hours)
    base_forecast = 150  # MW
    sample_forecast = []
    for i in range(6):
        # Simulate hourly variation
        if i in [1, 2]:  # Peak hours
            multiplier = 1.2
        elif i in [0, 5]:  # Lower demand
            multiplier = 0.9
        else:
            multiplier = 1.0
        sample_forecast.append(round(base_forecast * multiplier * (1 + 0.1 * np.random.rand()), 1))

    return jsonify({
        'cities': cities,
        'areas': areas,
        'sources': sources,
        'sample_forecast': sample_forecast,
        'sample_target_date': '2025-01-15'  # For similar days analysis
    })

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload and process CSV data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if file and file.filename.endswith('.csv'):
            # Read CSV data
            df = pd.read_csv(file)

            # Validate CSV format
            required_columns = ['Date', 'Hour', 'City', 'Area', 'Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals', 'Total']
            if not all(col in df.columns for col in required_columns):
                return jsonify({
                    'success': False,
                    'error': f'CSV must contain columns: {", ".join(required_columns)}'
                }), 400

            # Basic statistics
            stats = {
                'total_records': len(df),
                'cities': df['City'].nunique(),
                'areas': df['Area'].nunique(),
                'date_range': {
                    'start': df['Date'].min(),
                    'end': df['Date'].max()
                },
                'consumption_stats': {
                    'total_avg': float(df['Total'].mean()),
                    'total_max': float(df['Total'].max()),
                    'total_min': float(df['Total'].min()),
                    'industrial_avg': float(df['Industrial'].mean()),
                    'households_avg': float(df['Households'].mean()),
                    'schools_avg': float(df['Schools'].mean()),
                    'colleges_avg': float(df['Colleges'].mean()),
                    'hospitals_avg': float(df['Hospitals'].mean())
                }
            }

            # Save the uploaded file for processing
            upload_path = 'uploaded_energy_data.csv'
            df.to_csv(upload_path, index=False)

            return jsonify({
                'success': True,
                'message': f'CSV uploaded successfully. {len(df)} records processed.',
                'stats': stats,
                'filename': file.filename
            })
        else:
            return jsonify({'success': False, 'error': 'Please upload a CSV file'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing CSV: {str(e)}'}), 400

    # --- Hardcoded Users and Energy Balances ---

USERS = [
    {"user_id": "user1", "name": "Alice", "city": "Bangalore", "area": "MG Road", "balance_mw": 500.0},
    {"user_id": "user2", "name": "Bob", "city": "Bangalore", "area": "Indiranagar", "balance_mw": 300.0},
    {"user_id": "user3", "name": "Charlie", "city": "Bangalore", "area": "Koramangala", "balance_mw": 200.0},
    {"user_id": "user4", "name": "Diana", "city": "Bangalore", "area": "Malleshwaram", "balance_mw": 150.0},
]

def get_user_by_id(user_id: str):
    for user in USERS:
        if user["user_id"] == user_id:
            return user
    return None

@app.route('/api/users', methods=['GET'])
def api_retrieve_users():
    """Retrieve all users and their energy balances."""
    return jsonify({"users": USERS})

@app.route('/api/user-balance/<user_id>', methods=['GET'])
def api_get_user_balance(user_id):
    """Get the energy balance (MW) for a specific user."""
    user = get_user_by_id(user_id)
    if user:
        return jsonify({"user_id": user_id, "name": user["name"], "balance_mw": user["balance_mw"]})
    else:
        return jsonify({"error": f"User {user_id} not found."}), 404

@app.route('/api/trade-energy', methods=['POST'])
def api_trade_energy_units():
    """Trade energy units (MW) between hardcoded users. Updates balances and stores transaction in MongoDB."""
    try:
        data = request.json
        sender_id = data.get("sender_id")
        receiver_id = data.get("receiver_id")
        units_mw = float(data.get("units_mw", 0))
        note = data.get("note")
        sender = get_user_by_id(sender_id)
        receiver = get_user_by_id(receiver_id)
        if not sender or not receiver:
            return jsonify({
                "status": "failed",
                "message": "Sender or receiver not found.",
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "units_mw": units_mw
            }), 400
        if units_mw <= 0:
            return jsonify({
                "status": "failed",
                "message": "Units to trade must be positive.",
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "units_mw": units_mw
            }), 400
        if sender["balance_mw"] < units_mw:
            return jsonify({
                "status": "failed",
                "message": "Sender does not have enough balance.",
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "units_mw": units_mw
            }), 400
        # Update balances
        sender["balance_mw"] -= units_mw
        receiver["balance_mw"] += units_mw
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        transaction = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "units_mw": units_mw,
            "note": note or "Energy units traded successfully."
        }
        # Store transaction in MongoDB
        # transactions_collection.insert_one(transaction)
        return jsonify({
            "status": "success",
            **transaction,
            "sender_balance_mw": sender["balance_mw"],
            "receiver_balance_mw": receiver["balance_mw"]
        })
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 400

if __name__ == '__main__':
    # Create templates directory and HTML file if they don't exist
    os.makedirs('templates', exist_ok=True)

    # Create the updated HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grid Operations Dashboard - Enhanced</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }

        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .nav-tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 14px;
            font-weight: 600;
            color: #6c757d;
            transition: all 0.3s;
        }
        .nav-tab.active {
            background: white;
            color: #495057;
            border-bottom: 3px solid #007bff;
        }
        .nav-tab:hover:not(.active) {
            background: #e9ecef;
            color: #495057;
        }

        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }

        .tools-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            padding: 30px;
        }
        .tool-card {
            background: #f8f9ff;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e1e8ed;
        }
        .tool-card h3 {
            margin: 0 0 20px 0;
            color: #2c3e50;
            font-size: 1.4em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #34495e;
        }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            outline: none;
            border-color: #3498db;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        .btn:active { transform: translateY(0); }
        .btn-secondary {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
        }
        .btn-success {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }
        .btn-warning {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #e8f6f3;
            border-radius: 8px;
            border-left: 5px solid #27ae60;
        }
        .error {
            background: #fdf2f2;
            border-left: 5px solid #e74c3c;
            color: #c0392b;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }

        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            background: #f8f9ff;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e3f2fd;
        }
        .upload-area.dragover {
            border-color: #2980b9;
            background: #e3f2fd;
        }

        .sources-container { margin-top: 15px; }
        .source-row {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr auto;
            gap: 10px;
            align-items: end;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
        }
        .forecast-row {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-bottom: 10px;
        }
        .remove-btn {
            background: #e74c3c;
            padding: 8px 12px;
            font-size: 12px;
        }
        .add-btn {
            background: #27ae60;
            padding: 8px 16px;
            font-size: 14px;
            margin-top: 10px;
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 13px;
        }
        .plot-container { margin-top: 20px; text-align: center; }
        .plot-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .status-indicator {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-normal { background: #d5f4e6; color: #27ae60; }
        .status-warning { background: #fef9e7; color: #f39c12; }
        .status-critical { background: #fadbd8; color: #e74c3c; }

        .sector-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .sector-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .sector-card h5 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
        }
        .sector-card .value {
            font-size: 20px;
            font-weight: bold;
            color: #3498db;
        }
        .sector-card .unit {
            font-size: 12px;
            color: #7f8c8d;
        }

        .similar-day {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #3498db;
        }
        .similar-day h5 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .similar-day .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            font-size: 12px;
            color: #7f8c8d;
        }
        .user-section {
            margin: 30px 0;
            padding: 25px;
            background: #f8f9ff;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e1e8ed;
        }
        .user-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .user-card {
            background: white;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.07);
            border-left: 5px solid #3498db;
        }
        .user-card h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .user-card .balance {
            font-size: 22px;
            font-weight: bold;
            color: #27ae60;
        }
        .user-card .location {
            font-size: 13px;
            color: #7f8c8d;
        }
        .trade-form {
            background: #e8f6f3;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #27ae60;
        }
        .trade-form .form-row {
            grid-template-columns: 1fr 1fr 1fr;
        }
        .trade-form label {
            font-weight: 600;
            color: #34495e;
        }
        .trade-result {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Enhanced Grid Operations Dashboard</h1>
            <p>AI-Powered Electric Grid Management with RAG Pipeline & Sector Analysis</p>
        </div>

        <nav class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('forecasting')">🔮 Forecasting</button>
            <button class="nav-tab" onclick="switchTab('optimization')">⚖️ Optimization</button>
            <button class="nav-tab" onclick="switchTab('analysis')">📊 Analysis</button>
            <button class="nav-tab" onclick="switchTab('data-upload')">📁 Data Upload</button>
            <button class="nav-tab" onclick="switchTab('users')">👤 Users & Trading</button>
        </nav>

        <!-- Forecasting Tab -->
        <div id="forecasting" class="tab-content active">
            <div class="tools-grid">
                <!-- Consumption Prediction -->
                <div class="tool-card">
                    <h3>📊 Consumption Prediction</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>City:</label>
                            <select id="pred-city">
                                <option value="Bangalore">Bangalore</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Area (Optional):</label>
                            <select id="pred-area">
                                <option value="">All Areas</option>
                                <option value="MG Road">MG Road</option>
                                <option value="Brigade Road">Brigade Road</option>
                                <option value="Church Street">Church Street</option>
                                <option value="Richmond Town">Richmond Town</option>
                                <option value="Shivajinagar">Shivajinagar</option>
                                <option value="Vasanth Nagar">Vasanth Nagar</option>
                                <option value="Ulsoor (Halasuru)">Ulsoor (Halasuru)</option>
                                <option value="Indiranagar">Indiranagar</option>
                                <option value="Koramangala">Koramangala</option>
                                <option value="Malleshwaram">Malleshwaram</option>
                                <option value="Rajajinagar">Rajajinagar</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Forecast Horizon (hours):</label>
                            <input type="number" id="pred-horizon" value="24" min="1" max="168">
                        </div>
                        <div class="form-group">
                            <label>Weather Hint (optional):</label>
                            <input type="text" id="pred-weather" placeholder="e.g., Hot day expected, storm approaching">
                        </div>
                    </div>
                    <button class="btn" onclick="predictConsumption()">Generate Forecast</button>
                    <div id="pred-result"></div>
                </div>

                <!-- Energy Preparation -->
                <div class="tool-card">
                    <h3>🔋 3-Hour Energy Preparation</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>City:</label>
                            <select id="prep-city">
                                <option value="Bangalore">Bangalore</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Area (Optional):</label>
                            <select id="prep-area">
                                <option value="">All Areas</option>
                                <option value="MG Road">MG Road</option>
                                <option value="Brigade Road">Brigade Road</option>
                                <option value="Church Street">Church Street</option>
                                <option value="Richmond Town">Richmond Town</option>
                                <option value="Shivajinagar">Shivajinagar</option>
                                <option value="Vasanth Nagar">Vasanth Nagar</option>
                                <option value="Ulsoor (Halasuru)">Ulsoor (Halasuru)</option>
                                <option value="Indiranagar">Indiranagar</option>
                                <option value="Koramangala">Koramangala</option>
                                <option value="Malleshwaram">Malleshwaram</option>
                                <option value="Rajajinagar">Rajajinagar</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Current Load (MW):</label>
                            <input type="number" id="prep-load" value="150" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Reserve Margin (%):</label>
                            <input type="number" id="prep-margin" value="15" step="0.1">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Notes (optional):</label>
                        <textarea id="prep-notes" placeholder="e.g., Heat wave expected, maintenance scheduled"></textarea>
                    </div>
                    <button class="btn" onclick="prepEnergy()">Calculate Prep Capacity</button>
                    <div id="prep-result"></div>
                </div>
            </div>
        </div>

        <!-- Optimization Tab -->
        <div id="optimization" class="tab-content">
            <div class="tools-grid">
                <!-- Generation Monitoring -->
                <div class="tool-card">
                    <h3>🌱 Generation Monitoring</h3>
                    <div class="form-group">
                        <label>Monitoring Window (minutes):</label>
                        <input type="number" id="mon-window" value="15" min="1" max="1440">
                    </div>
                    <div class="form-group">
                        <label>Renewable Sources:</label>
                        <div id="sources-container" class="sources-container"></div>
                        <button class="btn add-btn" onclick="addSource()">+ Add Source</button>
                    </div>
                    <button class="btn btn-secondary" onclick="loadSampleData('sources')">Load Sample Sources</button>
                    <button class="btn" onclick="monitorGeneration()">Monitor Generation</button>
                    <div id="mon-result"></div>
                </div>

                <!-- Load Optimization -->
                <div class="tool-card">
                    <h3>⚖️ Load Optimization</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>City:</label>
                            <select id="opt-city">
                                <option value="Bangalore">Bangalore</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Area (Optional):</label>
                            <select id="opt-area">
                                <option value="">All Areas</option>
                                <option value="MG Road">MG Road</option>
                                <option value="Brigade Road">Brigade Road</option>
                                <option value="Church Street">Church Street</option>
                                <option value="Richmond Town">Richmond Town</option>
                                <option value="Shivajinagar">Shivajinagar</option>
                                <option value="Vasanth Nagar">Vasanth Nagar</option>
                                <option value="Ulsoor (Halasuru)">Ulsoor (Halasuru)</option>
                                <option value="Indiranagar">Indiranagar</option>
                                <option value="Koramangala">Koramangala</option>
                                <option value="Malleshwaram">Malleshwaram</option>
                                <option value="Rajajinagar">Rajajinagar</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Demand Forecast (MW) - Next 6 Hours:</label>
                        <div id="forecast-container" class="forecast-row">
                            <input type="number" placeholder="Hour 1" value="150">
                            <input type="number" placeholder="Hour 2" value="160">
                            <input type="number" placeholder="Hour 3" value="170">
                            <input type="number" placeholder="Hour 4" value="165">
                            <input type="number" placeholder="Hour 5" value="155">
                            <input type="number" placeholder="Hour 6" value="145">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Available Dispatchable Capacity (MW):</label>
                            <input type="number" id="opt-dispatch" value="140" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Current Renewables Output (MW):</label>
                            <input type="number" id="opt-renewables" value="30" step="0.1">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Constraints:</label>
                        <textarea id="opt-constraints" placeholder="e.g., Maintain 15% reserve, avoid residential curtailment">Maintain 15% reserve margin, avoid shedding if possible</textarea>
                    </div>
                    <button class="btn" onclick="optimizeLoad()">Optimize Load Balance</button>
                    <div id="opt-result"></div>
                </div>
            </div>
        </div>

        <!-- Analysis Tab -->
        <div id="analysis" class="tab-content">
            <div class="tools-grid">
                <!-- Consumption Pattern Analysis -->
                <div class="tool-card">
                    <h3>📈 Consumption Pattern Analysis</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>City:</label>
                            <select id="analysis-city">
                                <option value="Bangalore">Bangalore</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Area (Optional):</label>
                            <select id="analysis-area">
                                <option value="">All Areas</option>
                                <option value="MG Road">MG Road</option>
                                <option value="Brigade Road">Brigade Road</option>
                                <option value="Church Street">Church Street</option>
                                <option value="Richmond Town">Richmond Town</option>
                                <option value="Shivajinagar">Shivajinagar</option>
                                <option value="Vasanth Nagar">Vasanth Nagar</option>
                                <option value="Ulsoor (Halasuru)">Ulsoor (Halasuru)</option>
                                <option value="Indiranagar">Indiranagar</option>
                                <option value="Koramangala">Koramangala</option>
                                <option value="Malleshwaram">Malleshwaram</option>
                                <option value="Rajajinagar">Rajajinagar</option>
                            </select>
                        </div>
                    </div>
                    <button class="btn" onclick="analyzePatterns()">Analyze Consumption Patterns</button>
                    <div id="analysis-result"></div>
                </div>

                <!-- Similar Days Analysis -->
                <div class="tool-card">
                    <h3>📅 Similar Consumption Days</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>City:</label>
                            <select id="similar-city">
                                <option value="Bangalore">Bangalore</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Area (Optional):</label>
                            <select id="similar-area">
                                <option value="">All Areas</option>
                                <option value="MG Road">MG Road</option>
                                <option value="Brigade Road">Brigade Road</option>
                                <option value="Church Street">Church Street</option>
                                <option value="Richmond Town">Richmond Town</option>
                                <option value="Shivajinagar">Shivajinagar</option>
                                <option value="Vasanth Nagar">Vasanth Nagar</option>
                                <option value="Ulsoor (Halasuru)">Ulsoor (Halasuru)</option>
                                <option value="Indiranagar">Indiranagar</option>
                                <option value="Koramangala">Koramangala</option>
                                <option value="Malleshwaram">Malleshwaram</option>
                                <option value="Rajajinagar">Rajajinagar</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Target Date:</label>
                            <input type="date" id="similar-date" value="2025-01-15">
                        </div>
                        <div class="form-group">
                            <label>Number of Similar Days:</label>
                            <input type="number" id="similar-count" value="5" min="1" max="10">
                        </div>
                    </div>
                    <button class="btn btn-secondary" onclick="loadSampleData('similar')">Load Sample Date</button>
                    <button class="btn" onclick="findSimilarDays()">Find Similar Days</button>
                    <div id="similar-result"></div>
                </div>
            </div>
        </div>

        <!-- Data Upload Tab -->
        <div id="data-upload" class="tab-content">
            <div class="tools-grid">
                <div class="tool-card" style="grid-column: 1 / -1;">
                    <h3>📁 Data Upload & Management</h3>
                    <div class="upload-area" id="uploadArea">
                        <p><strong>📊 Upload Energy Consumption Data</strong></p>
                        <p>Drag and drop your CSV file here, or click to select</p>
                        <input type="file" id="csvFile" accept=".csv" style="display: none;">
                        <button class="btn btn-secondary" onclick="document.getElementById('csvFile').click()">Select CSV File</button>
                    </div>
                    <div class="form-group">
                        <label><strong>Expected CSV Format:</strong></label>
                        <p>Your CSV should contain the following columns:</p>
                        <code>Date, Hour, City, Area, Industrial, Households, Schools, Colleges, Hospitals, Total</code>
                        <p><strong>Example:</strong></p>
                        <pre>Date,Hour,City,Area,Industrial,Households,Schools,Colleges,Hospitals,Total,2025-01-01,00:00,Bangalore,MG Road,81.22,46.54,2.57,4.75,28.88,163.96</pre>
                    </div>
                    <div id="upload-result"></div>
                </div>
            </div>
        </div>

        <!-- Users & Trading Tab -->
        <div id="users" class="tab-content">
                    <div class="user-section">
                        <h3>👤 User Energy Balances & Trading</h3>
                        <div id="user-list" class="user-list"></div>
                        <div class="trade-form">
                            <h4>🔄 Trade Energy Units (MW)</h4>
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="trade-sender">Sender:</label>
                                    <select id="trade-sender"></select>
                                </div>
                                <div class="form-group">
                                    <label for="trade-receiver">Receiver:</label>
                                    <select id="trade-receiver"></select>
                                </div>
                                <div class="form-group">
                                    <label for="trade-units">Units (MW):</label>
                                    <input type="number" id="trade-units" min="0.1" step="0.1" value="10">
                                </div>
                            </div>
                            <div class="form-group">
                                <label for="trade-note">Note (optional):</label>
                                <input type="text" id="trade-note" placeholder="e.g., Peak hour transfer">
                            </div>
                            <button class="btn btn-success" onclick="tradeEnergy()">Execute Trade</button>
                            <div id="trade-result" class="trade-result"></div>
                        </div>
                    </div>
        </div>
    </div>

    <script>
        // Tab switching functionality
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });

            // Remove active class from all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab content
            document.getElementById(tabName).classList.add('active');

            // Add active class to clicked nav tab
            event.target.classList.add('active');
        }

        // Initialize with one source row
        addSource();

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const csvFile = document.getElementById('csvFile');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        csvFile.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        async function handleFileUpload(file) {
            const resultDiv = document.getElementById('upload-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Uploading and processing CSV...</div>';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await axios.post('/api/upload-csv', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });

                const result = response.data;
                if (result.success) {
                    const stats = result.stats;
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h4>✅ CSV Upload Successful</h4>
                            <p><strong>File:</strong> ${result.filename}</p>
                            <p><strong>Records:</strong> ${stats.total_records.toLocaleString()}</p>
                            <div class="sector-breakdown">
                                <div class="sector-card">
                                    <h5>Cities</h5>
                                    <div class="value">${stats.cities}</div>
                                </div>
                                <div class="sector-card">
                                    <h5>Areas</h5>
                                    <div class="value">${stats.areas}</div>
                                </div>
                                <div class="sector-card">
                                    <h5>Date Range</h5>
                                    <div class="value">${stats.date_range.start}</div>
                                    <div class="unit">to ${stats.date_range.end}</div>
                                </div>
                                <div class="sector-card">
                                    <h5>Avg Total</h5>
                                    <div class="value">${Math.round(stats.consumption_stats.total_avg)}</div>
                                    <div class="unit">MW</div>
                                </div>
                                <div class="sector-card">
                                    <h5>Peak Load</h5>
                                    <div class="value">${Math.round(stats.consumption_stats.total_max)}</div>
                                    <div class="unit">MW</div>
                                </div>
                            </div>
                            <p><em>✨ Data has been processed and is now available for RAG-powered analysis!</em></p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result error"><strong>❌ Upload Error:</strong> ${result.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Upload Failed:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        async function loadSampleData(type) {
            try {
                const response = await axios.get('/api/sample-data');
                const data = response.data;

                if (type === 'sources') {
                    // Clear existing sources
                    document.getElementById('sources-container').innerHTML = '';
                    // Add sample sources
                    data.sources.forEach(source => {
                        addSource(source);
                    });
                } else if (type === 'similar') {
                    document.getElementById('similar-date').value = data.sample_target_date;
                }
            } catch (error) {
                console.error('Error loading sample data:', error);
            }
        }

        function addSource(data = null) {
            const container = document.getElementById('sources-container');
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-row';
            sourceDiv.innerHTML = `
                <input type="text" placeholder="Source Name" value="${data?.name || ''}" class="source-name">
                <select class="source-kind">
                    <option value="solar" ${data?.kind === 'solar' ? 'selected' : ''}>Solar</option>
                    <option value="wind" ${data?.kind === 'wind' ? 'selected' : ''}>Wind</option>
                    <option value="hydro" ${data?.kind === 'hydro' ? 'selected' : ''}>Hydro</option>
                    <option value="rooftop" ${data?.kind === 'rooftop' ? 'selected' : ''}>Rooftop</option>
                </select>
                <input type="number" placeholder="Current MW" value="${data?.current_mw || ''}" step="0.1" class="source-current">
                <input type="number" placeholder="Expected MW" value="${data?.expected_mw || ''}" step="0.1" class="source-expected">
                <button class="btn remove-btn" onclick="this.parentElement.remove()">Remove</button>
            `;
            container.appendChild(sourceDiv);
        }

        async function predictConsumption() {
        const resultDiv = document.getElementById('pred-result');
        resultDiv.innerHTML = '<div class="loading">🔄 Generating RAG-powered forecast...</div>';

        try {
            const response = await axios.post('/api/predict', {
                city: document.getElementById('pred-city').value,
                area: document.getElementById('pred-area').value || null,
                horizon_hours: parseInt(document.getElementById('pred-horizon').value),
                weather_hint: document.getElementById('pred-weather').value
            });

            const result = response.data.data;

            // 🔽 THIS is the part you pasted
            breakdownHtml = '';
            if (result.breakdown) {
                breakdownHtml = '<h5>Sector Breakdown (Hour 1):</h5><div class="sector-breakdown">';
                const sectors = ['Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals'];
                sectors.forEach(sector => {
                    if (result.breakdown[0] && result.breakdown[0][sector]) {
                        breakdownHtml += `
                            <div class="sector-card">
                                <h5>${sector}</h5>
                                <div class="value">${Math.round(result.breakdown[0][sector])}</div>
                                <div class="unit">MW</div>
                            </div>
                        `;
                    }
                });
                breakdownHtml += '</div>';
            }

            resultDiv.innerHTML = `
                <div class="result">
                    <h4>✅ AI-Powered Forecast Generated</h4>
                    <p><strong>Location:</strong> ${result.city}${result.area ? ` - ${result.area}` : ''}</p>
                    <p><strong>Horizon:</strong> ${result.horizon_hours} hours</p>
                    <p><strong>Data Points Used:</strong> ${result.data_points_used}</p>
                    <p><strong>Next 6 Hours:</strong> ${result.forecast_mw.slice(0,6).map(x => Math.round(x)).join(', ')} MW</p>
                    ${breakdownHtml}
                    <div class="plot-container">
                        <img src="data:image/png;base64,${result.plot_base64}" alt="Forecast Plot">
                    </div>
                </div>
            `;
        } catch (error) {
            resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
        }
    }

        async function findSimilarDays() {
            const resultDiv = document.getElementById('similar-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Finding similar consumption days using RAG...</div>';

            try {
                const response = await axios.post('/api/similar-days', {
                    city: document.getElementById('similar-city').value,
                    area: document.getElementById('similar-area').value || null,
                    target_date: document.getElementById('similar-date').value,
                    n_days: parseInt(document.getElementById('similar-count').value)
                });

                const result = response.data.data;

                if (result.error) {
                    resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${result.error}</div>`;
                    return;
                }

                let similarDaysHtml = '';
                if (result.similar_days && result.similar_days.length > 0) {
                    similarDaysHtml = '<h5>📅 Similar Days Found:</h5>';
                    result.similar_days.forEach(day => {
                        similarDaysHtml += `
                            <div class="similar-day">
                                <h5>📆 ${day.date}</h5>
                                <p><strong>Total Consumption:</strong> ${Math.round(day.total_consumption)} MW</p>
                                <p><strong>Peak:</strong> ${Math.round(day.peak_value)} MW at ${day.peak_hour}</p>
                                <div class="stats">
                                    <span><strong>Industrial:</strong> ${Math.round(day.sector_breakdown.Industrial)} MW</span>
                                    <span><strong>Households:</strong> ${Math.round(day.sector_breakdown.Households)} MW</span>
                                    <span><strong>Commercial:</strong> ${Math.round(day.sector_breakdown.Schools + day.sector_breakdown.Colleges)} MW</span>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    similarDaysHtml = '<p><em>No similar days found in the dataset.</em></p>';
                }

                resultDiv.innerHTML = `
                    <div class="result">
                        <h4>✅ Similar Days Analysis Complete</h4>
                        <p><strong>Target Date:</strong> ${result.target_date}</p>
                        <p><strong>Location:</strong> ${result.city}${result.area ? ` - ${result.area}` : ''}</p>
                        <p><strong>Target Total Consumption:</strong> ${Math.round(result.target_total_consumption)} MW</p>
                        ${similarDaysHtml}
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        async function prepEnergy() {
            const resultDiv = document.getElementById('prep-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Calculating preparation capacity with RAG context...</div>';

            try {
                const response = await axios.post('/api/prep', {
                    city: document.getElementById('prep-city').value,
                    area: document.getElementById('prep-area').value || null,
                    current_load_mw: parseFloat(document.getElementById('prep-load').value),
                    reserve_margin_pct: parseFloat(document.getElementById('prep-margin').value),
                    notes: document.getElementById('prep-notes').value
                });

                const result = response.data.data;

                let sectorBreakdownHtml = '';
                if (result.sector_breakdown && result.sector_breakdown.length > 0) {
                    sectorBreakdownHtml = '<h5>Sector-wise Preparation (Hour 1):</h5><div class="sector-breakdown">';
                    const breakdown = result.sector_breakdown[0];
                    Object.keys(breakdown).forEach(sector => {
                        sectorBreakdownHtml += `
                            <div class="sector-card">
                                <h5>${sector}</h5>
                                <div class="value">${Math.round(breakdown[sector])}</div>
                                <div class="unit">MW</div>
                            </div>
                        `;
                    });
                    sectorBreakdownHtml += '</div>';
                }

                resultDiv.innerHTML = `
                    <div class="result">
                        <h4>✅ Preparation Plan Ready</h4>
                        <p><strong>Location:</strong> ${result.city}${result.area ? ` - ${result.area}` : ''}</p>
                        <p><strong>Recommended Capacity:</strong></p>
                        <ul>
                            <li>Hour 1: ${Math.round(result.prep_mw[0])} MW</li>
                            <li>Hour 2: ${Math.round(result.prep_mw[1])} MW</li>
                            <li>Hour 3: ${Math.round(result.prep_mw[2])} MW</li>
                        </ul>
                        ${sectorBreakdownHtml}
                        <p><strong>Rationale:</strong> ${result.rationale}</p>
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        async function monitorGeneration() {
            const resultDiv = document.getElementById('mon-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Monitoring generation sources...</div>';

            try {
                // Collect sources data
                const sourceRows = document.querySelectorAll('.source-row');
                const sources = Array.from(sourceRows).map(row => ({
                    name: row.querySelector('.source-name').value,
                    kind: row.querySelector('.source-kind').value,
                    current_mw: parseFloat(row.querySelector('.source-current').value),
                    expected_mw: parseFloat(row.querySelector('.source-expected').value) || null
                })).filter(s => s.name && !isNaN(s.current_mw));

                if (sources.length === 0) {
                    resultDiv.innerHTML = '<div class="result error"><strong>❌ Error:</strong> Please add at least one valid source</div>';
                    return;
                }

                const response = await axios.post('/api/monitor', {
                    window_minutes: parseInt(document.getElementById('mon-window').value),
                    sources: sources
                });

                const result = response.data.data;
                const statusClass = `status-${result.status}`;

                let anomaliesHtml = '';
                if (result.anomalies.length > 0) {
                    anomaliesHtml = '<h5>🚨 Anomalies Detected:</h5><ul>';
                    result.anomalies.forEach(anomaly => {
                        anomaliesHtml += `<li><strong>${anomaly.name}:</strong> ${anomaly.issue} - <em>${anomaly.suggested_action}</em></li>`;
                    });
                    anomaliesHtml += '</ul>';
                }

                resultDiv.innerHTML = `
                    <div class="result">
                        <h4>✅ Generation Monitoring Complete</h4>
                        <p><strong>Status:</strong> <span class="status-indicator ${statusClass}">${result.status}</span></p>
                        <p><strong>Total Current Output:</strong> ${Math.round(result.total_now_mw)} MW</p>
                        <p><strong>Total Expected Output:</strong> ${Math.round(result.total_expected_mw)} MW</p>
                        <p><strong>Summary:</strong> ${result.summary}</p>
                        ${anomaliesHtml}
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        async function optimizeLoad() {
            const resultDiv = document.getElementById('opt-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Optimizing load balance with sector analysis...</div>';

            try {
                // Collect forecast values
                const forecastInputs = document.querySelectorAll('#forecast-container input');
                const demandForecast = Array.from(forecastInputs).map(input => parseFloat(input.value)).filter(val => !isNaN(val));

                if (demandForecast.length < 3) {
                    resultDiv.innerHTML = '<div class="result error"><strong>❌ Error:</strong> Please provide at least 3 forecast values</div>';
                    return;
                }

                const response = await axios.post('/api/optimize', {
                    city: document.getElementById('opt-city').value,
                    area: document.getElementById('opt-area').value || null,
                    demand_forecast_mw: demandForecast,
                    dispatchable_mw: parseFloat(document.getElementById('opt-dispatch').value),
                    renewables_now_mw: parseFloat(document.getElementById('opt-renewables').value),
                    constraints: document.getElementById('opt-constraints').value
                });

                const result = response.data.data;
                const riskClass = result.residual_risk === 'high' ? 'status-critical' :
                                 result.residual_risk === 'medium' ? 'status-warning' : 'status-normal';

                let actionsHtml = '<h5>📋 Recommended Actions:</h5><ol>';
                result.actions.forEach(action => {
                    const costColor = action.cost_level === 'high' ? '#e74c3c' :
                                    action.cost_level === 'medium' ? '#f39c12' : '#27ae60';
                    actionsHtml += `
                        <li>
                            <strong>${action.title}</strong><br>
                            <small style="color: ${costColor}">💰 ${action.cost_level.toUpperCase()} cost</small> |
                            <small>⚡ ${Math.round(action.mw_effect)} MW impact</small> |
                            <small>⏱️ ${action.eta_min} min ETA</small> |
                            <small>🏢 ${action.sector || 'General'} sector</small><br>
                            <em>${action.notes}</em>
                        </li>
                    `;
                });
                actionsHtml += '</ol>';

                resultDiv.innerHTML = `
                    <div class="result">
                        <h4>✅ Load Optimization Complete</h4>
                        <p><strong>Location:</strong> ${result.city}${result.area ? ` - ${result.area}` : ''}</p>
                        <p><strong>Expected Relief:</strong> ${Math.round(result.expected_net_relief_mw)} MW</p>
                        <p><strong>Residual Risk:</strong> <span class="status-indicator ${riskClass}">${result.residual_risk}</span></p>
                        ${actionsHtml}
                        <p><strong>Rationale:</strong> ${result.rationale}</p>
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        async function analyzePatterns() {
            const resultDiv = document.getElementById('analysis-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Analyzing consumption patterns with RAG...</div>';

            try {
                const response = await axios.post('/api/analyze-patterns', {
                    city: document.getElementById('analysis-city').value,
                    area: document.getElementById('analysis-area').value || null
                });

                const result = response.data.data;
                const stats = result.statistical_analysis;
                const insights = result.ai_insights;

                let statsHtml = '';
                if (stats) {
                    statsHtml = '<h5>📊 Statistical Analysis:</h5><div class="sector-breakdown">';
                    Object.keys(stats.average_by_sector || {}).forEach(sector => {
                        statsHtml += `
                            <div class="sector-card">
                                <h5>${sector}</h5>
                                <div class="value">${Math.round(stats.average_by_sector[sector])}</div>
                                <div class="unit">MW avg</div>
                            </div>
                        `;
                    });
                    statsHtml += '</div>';

                    statsHtml += `
                        <p><strong>Peak Consumption:</strong> ${Math.round(stats.peak_consumption.value)} MW at ${stats.peak_consumption.time}</p>
                        <p><strong>Low Consumption:</strong> ${Math.round(stats.low_consumption.value)} MW at ${stats.low_consumption.time}</p>
                    `;
                }

                let insightsHtml = '';
                if (insights && insights.insights) {
                    insightsHtml = '<h5>🧠 AI Insights:</h5><ul>';
                    insights.insights.forEach(insight => {
                        insightsHtml += `<li>${insight}</li>`;
                    });
                    insightsHtml += '</ul>';
                }

                let recommendationsHtml = '';
                if (insights && insights.recommendations) {
                    recommendationsHtml = '<h5>💡 Recommendations:</h5><ul>';
                    insights.recommendations.forEach(rec => {
                        recommendationsHtml += `<li>${rec}</li>`;
                    });
                    recommendationsHtml += '</ul>';
                }

                resultDiv.innerHTML = `
                    <div class="result">
                        <h4>✅ Consumption Pattern Analysis Complete</h4>
                        <p><strong>Location:</strong> ${result.city}${result.area ? ` - ${result.area}` : ''}</p>
                        ${statsHtml}
                        ${insightsHtml}
                        ${recommendationsHtml}
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }
        // --- USER & ENERGY TRADING SECTION ---
        let usersData = [];

        async function loadUsers() {
            try {
                const response = await axios.get('/api/users');
                usersData = response.data.users;
                renderUserList();
                populateTradeDropdowns();
            } catch (error) {
                document.getElementById('user-list').innerHTML = `<div class="result error"><strong>❌ Error loading users:</strong> ${error.response?.data?.error || error.message}</div>`;
            }
        }

        function renderUserList() {
            const userListDiv = document.getElementById('user-list');
            if (!usersData || usersData.length === 0) {
                userListDiv.innerHTML = '<p>No users found.</p>';
                return;
            }
            userListDiv.innerHTML = usersData.map(user => `
                <div class="user-card">
                    <h4>${user.name}</h4>
                    <div class="balance">${Math.round(user.balance_mw * 100) / 100} MW</div>
                    <div class="location">${user.city} - ${user.area}</div>
                    <div style="font-size:12px;color:#888;">ID: ${user.user_id}</div>
                </div>
            `).join('');
        }

        function populateTradeDropdowns() {
            const senderSelect = document.getElementById('trade-sender');
            const receiverSelect = document.getElementById('trade-receiver');
            senderSelect.innerHTML = '';
            receiverSelect.innerHTML = '';
            usersData.forEach(user => {
                senderSelect.innerHTML += `<option value="${user.user_id}">${user.name} (${user.area})</option>`;
                receiverSelect.innerHTML += `<option value="${user.user_id}">${user.name} (${user.area})</option>`;
            });
        }

        async function tradeEnergy() {
            const senderId = document.getElementById('trade-sender').value;
            const receiverId = document.getElementById('trade-receiver').value;
            const unitsMw = parseFloat(document.getElementById('trade-units').value);
            const note = document.getElementById('trade-note').value;
            const resultDiv = document.getElementById('trade-result');
            resultDiv.innerHTML = '<div class="loading">🔄 Executing trade...</div>';

            if (senderId === receiverId) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> Sender and receiver must be different users.</div>`;
                return;
            }
            if (isNaN(unitsMw) || unitsMw <= 0) {
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Error:</strong> Units to trade must be positive.</div>`;
                return;
            }

            try {
                const response = await axios.post('/api/trade-energy', {
                    sender_id: senderId,
                    receiver_id: receiverId,
                    units_mw: unitsMw,
                    note: note
                });
                const result = response.data;
                if (result.status === "success") {
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h4>✅ Trade Successful</h4>
                            <p><strong>Transaction ID:</strong> ${result.transaction_id}</p>
                            <p><strong>Timestamp:</strong> ${result.timestamp}</p>
                            <p><strong>Sender:</strong> ${result.sender_id} | <strong>New Balance:</strong> ${Math.round(result.sender_balance_mw * 100) / 100} MW</p>
                            <p><strong>Receiver:</strong> ${result.receiver_id} | <strong>New Balance:</strong> ${Math.round(result.receiver_balance_mw * 100) / 100} MW</p>
                            <p><strong>Units Traded:</strong> ${result.units_mw} MW</p>
                            <p><strong>Note:</strong> ${result.note}</p>
                        </div>
                    `;
                    // Refresh user list to show updated balances
                    await loadUsers();
                } else {
                    resultDiv.innerHTML = `<div class="result error"><strong>❌ Trade Failed:</strong> ${result.message}</div>`;
                }
            } catch (error) {
                const errMsg = error.response?.data?.message || error.message;
                resultDiv.innerHTML = `<div class="result error"><strong>❌ Trade Failed:</strong> ${errMsg}</div>`;
            }
        }

        // Load users when the tab is shown
        document.querySelector('button.nav-tab[onclick*="users"]').addEventListener('click', loadUsers);

        // Initial load for users tab if it's default
        if (document.getElementById('users').classList.contains('active')) {
            loadUsers();
        }
    </script>
</body>
</html>
'''

    # Write the template file
    with open('templates/index.html', 'w') as f:
        f.write(html_template)

    print("Starting Enhanced Grid Operations Flask Server...")
    print("Dashboard available at: http://localhost:5000")
    print("API endpoints:")
    print("   - /api/predict (consumption prediction)")
    print("   - /api/prep (3-hour energy preparation)")
    print("   - /api/monitor (generation monitoring)")
    print("   - /api/optimize (load optimization)")
    print("   - /api/analyze-patterns (consumption pattern analysis)")
    print("   - /api/similar-days (similar consumption days)")
    print("   - /api/upload-csv (CSV data upload)")
    print("   - /api/users (user list)")
    print("   - /api/user-balance/<user_id> (user balance)")
    print("   - /api/trade-energy (energy trading)")
    print("RAG pipeline with ChromaDB enabled for enhanced AI analysis")

    app.run(debug=True, host='0.0.0.0', port=5000)
