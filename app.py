"""
CyberShield - Threat Detection & Security Analysis Dashboard
=============================================================
Full-stack cybersecurity application with ML-powered URL threat detection,
real-time analytics, and comprehensive security monitoring.
"""

import os
import json
import csv
import io
import sqlite3
from datetime import datetime, timedelta
import random

import numpy as np
import joblib
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, Response, g
)
from werkzeug.security import generate_password_hash, check_password_hash

# Import ML module
from model.train_model import extract_features, FEATURE_NAMES, train as train_model

# ──────────────────────────────────────────────────────────
# APP CONFIGURATION
# ──────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'cybershield_secret_2026_x9k2m1'
app.config['DATABASE'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cybershield.db'
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# ──────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────

def get_db():
    """Get database connection for current request."""
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Close database at end of request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database schema."""
    db = sqlite3.connect(app.config['DATABASE'])
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_score REAL NOT NULL,
            confidence REAL NOT NULL,
            is_anomaly INTEGER DEFAULT 0,
            features TEXT,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_scans INTEGER DEFAULT 0,
            safe_count INTEGER DEFAULT 0,
            suspicious_count INTEGER DEFAULT 0,
            malicious_count INTEGER DEFAULT 0
        )
    ''')
    db.commit()
    db.close()


# ──────────────────────────────────────────────────────────
# ML MODEL LOADING
# ──────────────────────────────────────────────────────────

def load_models():
    """Load trained ML models."""
    models = {}
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    lr_path = os.path.join(MODEL_DIR, 'lr_model.pkl')
    iso_path = os.path.join(MODEL_DIR, 'iso_model.pkl')
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')

    if os.path.exists(model_path):
        models['rf'] = joblib.load(model_path)
    if os.path.exists(lr_path):
        models['lr'] = joblib.load(lr_path)
    if os.path.exists(iso_path):
        models['iso'] = joblib.load(iso_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            models['metrics'] = json.load(f)

    return models


# ──────────────────────────────────────────────────────────
# AUTH HELPERS
# ──────────────────────────────────────────────────────────

def login_required(f):
    """Decorator to require login."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────────────────
# ROUTES: AUTHENTICATION
# ──────────────────────────────────────────────────────────

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')

        db = get_db()
        existing = db.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if existing:
            flash('Username or email already exists.', 'error')
            return render_template('signup.html')

        password_hash = generate_password_hash(password)
        db.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        db.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ──────────────────────────────────────────────────────────
# ROUTES: MAIN PAGES
# ──────────────────────────────────────────────────────────

@app.route('/')
@login_required
def home():
    return render_template('home.html', username=session.get('username'))


@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html', username=session.get('username'))


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', username=session.get('username'))


@app.route('/about')
@login_required
def about():
    return render_template('about.html', username=session.get('username'))


# ──────────────────────────────────────────────────────────
# API: SCAN URL
# ──────────────────────────────────────────────────────────

@app.route('/api/scan', methods=['POST'])
@login_required
def scan_url():
    """Scan a URL using the trained ML model."""
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    models = load_models()
    if 'rf' not in models:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 500

    # Extract features
    features = extract_features(url)
    feature_vector = np.array([[features[name] for name in FEATURE_NAMES]])

    # Random Forest prediction
    rf_prediction = models['rf'].predict(feature_vector)[0]
    rf_proba = models['rf'].predict_proba(feature_vector)[0]
    confidence = float(max(rf_proba)) * 100

    # Anomaly detection
    is_anomaly = 0
    if 'iso' in models:
        iso_pred = models['iso'].predict(feature_vector)[0]
        is_anomaly = 1 if iso_pred == -1 else 0

    # Determine risk level and score
    malicious_prob = float(rf_proba[1]) * 100
    if malicious_prob >= 70:
        prediction = 'malicious'
        risk_score = min(100, malicious_prob + random.uniform(0, 5))
    elif malicious_prob >= 35:
        prediction = 'suspicious'
        risk_score = malicious_prob + random.uniform(-5, 5)
    else:
        prediction = 'safe'
        risk_score = max(0, malicious_prob - random.uniform(0, 3))

    risk_score = round(min(100, max(0, risk_score)), 1)
    confidence = round(confidence, 1)

    # Save to database
    db = get_db()
    db.execute('''
        INSERT INTO scan_logs (user_id, url, prediction, risk_score, confidence, is_anomaly, features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        session['user_id'], url, prediction, risk_score, confidence,
        is_anomaly, json.dumps(features)
    ))

    # Update daily stats
    today = datetime.now().strftime('%Y-%m-%d')
    existing = db.execute(
        'SELECT * FROM daily_stats WHERE date = ?', (today,)
    ).fetchone()

    if existing:
        col = f'{prediction}_count'
        db.execute(f'''
            UPDATE daily_stats
            SET total_scans = total_scans + 1, {col} = {col} + 1
            WHERE date = ?
        ''', (today,))
    else:
        db.execute('''
            INSERT INTO daily_stats (date, total_scans, safe_count, suspicious_count, malicious_count)
            VALUES (?, 1, ?, ?, ?)
        ''', (today,
              1 if prediction == 'safe' else 0,
              1 if prediction == 'suspicious' else 0,
              1 if prediction == 'malicious' else 0))

    db.commit()

    return jsonify({
        'url': url,
        'prediction': prediction,
        'risk_score': risk_score,
        'confidence': confidence,
        'is_anomaly': bool(is_anomaly),
        'features': features,
        'malicious_probability': round(malicious_prob, 2)
    })


# ──────────────────────────────────────────────────────────
# API: LOGS
# ──────────────────────────────────────────────────────────

@app.route('/api/logs')
@login_required
def get_logs():
    """Get scan logs for current user."""
    db = get_db()
    logs = db.execute(
        '''SELECT id, url, prediction, risk_score, confidence, is_anomaly, scanned_at
           FROM scan_logs WHERE user_id = ? ORDER BY scanned_at DESC''',
        (session['user_id'],)
    ).fetchall()

    return jsonify([{
        'id': log['id'],
        'url': log['url'],
        'prediction': log['prediction'],
        'risk_score': log['risk_score'],
        'confidence': log['confidence'],
        'is_anomaly': bool(log['is_anomaly']),
        'scanned_at': log['scanned_at']
    } for log in logs])


@app.route('/api/logs/<int:log_id>', methods=['DELETE'])
@login_required
def delete_log(log_id):
    """Delete a single log entry."""
    db = get_db()
    db.execute(
        'DELETE FROM scan_logs WHERE id = ? AND user_id = ?',
        (log_id, session['user_id'])
    )
    db.commit()
    return jsonify({'success': True})


@app.route('/api/logs/clear', methods=['DELETE'])
@login_required
def clear_logs():
    """Clear all logs for current user."""
    db = get_db()
    db.execute('DELETE FROM scan_logs WHERE user_id = ?', (session['user_id'],))
    db.commit()
    return jsonify({'success': True})


@app.route('/api/logs/export')
@login_required
def export_logs():
    """Export logs as CSV."""
    db = get_db()
    logs = db.execute(
        '''SELECT url, prediction, risk_score, confidence, is_anomaly, scanned_at
           FROM scan_logs WHERE user_id = ? ORDER BY scanned_at DESC''',
        (session['user_id'],)
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['URL', 'Prediction', 'Risk Score', 'Confidence', 'Anomaly', 'Timestamp'])
    for log in logs:
        writer.writerow([
            log['url'], log['prediction'], log['risk_score'],
            log['confidence'], 'Yes' if log['is_anomaly'] else 'No',
            log['scanned_at']
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=cybershield_logs.csv'}
    )


# ──────────────────────────────────────────────────────────
# API: ANALYTICS
# ──────────────────────────────────────────────────────────

@app.route('/api/analytics/overview')
@login_required
def analytics_overview():
    """Get overview statistics."""
    db = get_db()

    total = db.execute(
        'SELECT COUNT(*) as c FROM scan_logs WHERE user_id = ?',
        (session['user_id'],)
    ).fetchone()['c']

    safe = db.execute(
        "SELECT COUNT(*) as c FROM scan_logs WHERE user_id = ? AND prediction = 'safe'",
        (session['user_id'],)
    ).fetchone()['c']

    suspicious = db.execute(
        "SELECT COUNT(*) as c FROM scan_logs WHERE user_id = ? AND prediction = 'suspicious'",
        (session['user_id'],)
    ).fetchone()['c']

    malicious = db.execute(
        "SELECT COUNT(*) as c FROM scan_logs WHERE user_id = ? AND prediction = 'malicious'",
        (session['user_id'],)
    ).fetchone()['c']

    anomalies = db.execute(
        'SELECT COUNT(*) as c FROM scan_logs WHERE user_id = ? AND is_anomaly = 1',
        (session['user_id'],)
    ).fetchone()['c']

    return jsonify({
        'total_scans': total,
        'safe': safe,
        'suspicious': suspicious,
        'malicious': malicious,
        'anomalies': anomalies
    })


@app.route('/api/analytics/daily')
@login_required
def analytics_daily():
    """Get daily scan statistics for the past 30 days."""
    db = get_db()

    # Generate last 30 days
    dates = []
    for i in range(29, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        dates.append(d)

    results = []
    for d in dates:
        row = db.execute(
            '''SELECT COALESCE(SUM(CASE WHEN prediction='safe' THEN 1 ELSE 0 END), 0) as safe,
                      COALESCE(SUM(CASE WHEN prediction='suspicious' THEN 1 ELSE 0 END), 0) as suspicious,
                      COALESCE(SUM(CASE WHEN prediction='malicious' THEN 1 ELSE 0 END), 0) as malicious,
                      COUNT(*) as total
               FROM scan_logs
               WHERE user_id = ? AND DATE(scanned_at) = ?''',
            (session['user_id'], d)
        ).fetchone()

        results.append({
            'date': d,
            'safe': row['safe'],
            'suspicious': row['suspicious'],
            'malicious': row['malicious'],
            'total': row['total']
        })

    return jsonify(results)


@app.route('/api/analytics/model')
@login_required
def analytics_model():
    """Get model performance metrics."""
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No metrics available'}), 404


@app.route('/api/train', methods=['POST'])
@login_required
def retrain_model():
    """Retrain the ML model."""
    try:
        metrics = train_model()
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()

    # Train model if not exists
    if not os.path.exists(os.path.join(MODEL_DIR, 'model.pkl')):
        print("[*] No trained model found. Training now...")
        train_model()

    app.run(debug=True, port=5000)
