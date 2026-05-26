"""
CyberShield ML Model Training Module
=====================================
Trains a Random Forest classifier on URL features to detect malicious URLs.
Also trains an Isolation Forest for anomaly detection.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    'free', 'win', 'winner', 'prize', 'claim', 'urgent', 'verify',
    'update', 'secure', 'login', 'account', 'suspend', 'alert',
    'confirm', 'restore', 'hack', 'crack', 'exploit', 'attack',
    'malware', 'phishing', 'trojan', 'ransomware', 'keylogger',
    'payload', 'backdoor', 'inject', 'steal', 'compromised',
    'download', 'install', 'exe', 'gift', 'offer', 'deal',
    'cheap', 'password', 'ssn', 'bank', 'paypal', 'bitcoin',
    'crypto', 'forex', 'pill', 'pharmacy', 'dating', 'adult',
    'generator', 'booster', 'followers', 'robux', 'vbucks',
    'scam', 'fraud', 'invoice', 'payment', 'wire', 'transfer'
]

SUSPICIOUS_TLDS = [
    '.tk', '.ml', '.cf', '.ga', '.xyz', '.info', '.top',
    '.click', '.link', '.work', '.date', '.racing', '.download',
    '.stream', '.gdn', '.bid', '.trade', '.webcam', '.party'
]


def extract_features(url):
    """Extract numerical features from a URL for ML prediction."""
    features = {}

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None

    # 1. URL length
    features['url_length'] = len(url)

    # 2. Number of dots
    features['num_dots'] = url.count('.')

    # 3. Number of hyphens
    features['num_hyphens'] = url.count('-')

    # 4. Number of underscores
    features['num_underscores'] = url.count('_')

    # 5. Number of slashes
    features['num_slashes'] = url.count('/')

    # 6. Number of question marks
    features['num_question_marks'] = url.count('?')

    # 7. Number of ampersands
    features['num_ampersands'] = url.count('&')

    # 8. Number of equals signs
    features['num_equals'] = url.count('=')

    # 9. Number of '@' symbols (common in phishing)
    features['num_at_symbols'] = url.count('@')

    # 10. HTTPS usage (1 = https, 0 = http)
    features['is_https'] = 1 if url.startswith('https') else 0

    # 11. Has IP address instead of domain
    features['has_ip_address'] = 1 if re.search(
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url
    ) else 0

    # 12. Domain length
    if parsed and parsed.netloc:
        features['domain_length'] = len(parsed.netloc)
    else:
        features['domain_length'] = 0

    # 13. Path length
    if parsed and parsed.path:
        features['path_length'] = len(parsed.path)
    else:
        features['path_length'] = 0

    # 14. Number of subdomains
    if parsed and parsed.netloc:
        features['num_subdomains'] = len(parsed.netloc.split('.')) - 2
    else:
        features['num_subdomains'] = 0

    # 15. Contains suspicious keywords
    url_lower = url.lower()
    features['num_suspicious_keywords'] = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower
    )

    # 16. Has suspicious TLD
    features['has_suspicious_tld'] = 1 if any(
        url_lower.endswith(tld) or tld + '/' in url_lower
        for tld in SUSPICIOUS_TLDS
    ) else 0

    # 17. Number of digits in URL
    features['num_digits'] = sum(c.isdigit() for c in url)

    # 18. Number of special characters
    features['num_special_chars'] = sum(
        not c.isalnum() and c not in ':/.-_' for c in url
    )

    # 19. URL entropy (randomness measure)
    from collections import Counter
    char_counts = Counter(url)
    url_len = len(url) if len(url) > 0 else 1
    features['url_entropy'] = -sum(
        (count / url_len) * np.log2(count / url_len)
        for count in char_counts.values()
    )

    # 20. Ratio of digits to total length
    features['digit_ratio'] = features['num_digits'] / max(features['url_length'], 1)

    return features


FEATURE_NAMES = [
    'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
    'num_slashes', 'num_question_marks', 'num_ampersands', 'num_equals',
    'num_at_symbols', 'is_https', 'has_ip_address', 'domain_length',
    'path_length', 'num_subdomains', 'num_suspicious_keywords',
    'has_suspicious_tld', 'num_digits', 'num_special_chars',
    'url_entropy', 'digit_ratio'
]


def train():
    """Train the ML model and save artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'urls.csv')
    model_dir = os.path.join(base_dir, 'model')

    # Load dataset
    print("[*] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"    Dataset size: {len(df)} URLs")
    print(f"    Labels: {df['label'].value_counts().to_dict()}")

    # Extract features
    print("[*] Extracting features...")
    features_list = []
    for url in df['url']:
        feat = extract_features(url)
        features_list.append([feat[name] for name in FEATURE_NAMES])

    X = np.array(features_list)
    y = (df['label'] == 'malicious').astype(int).values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── Train Random Forest ──
    print("[*] Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
    rf_recall = recall_score(y_test, y_pred_rf, zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
    rf_auc = roc_auc_score(y_test, y_proba_rf)

    print(f"    Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"    Precision: {rf_precision:.4f}")
    print(f"    Recall: {rf_recall:.4f}")
    print(f"    F1 Score: {rf_f1:.4f}")
    print(f"    AUC-ROC: {rf_auc:.4f}")

    # ── Train Logistic Regression (for comparison) ──
    print("[*] Training Logistic Regression model...")
    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    print(f"    Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # ── Train Isolation Forest (Anomaly Detection) ──
    print("[*] Training Isolation Forest for anomaly detection...")
    iso_model = IsolationForest(
        n_estimators=100, contamination=0.3,
        random_state=42, n_jobs=-1
    )
    iso_model.fit(X_train)

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred_rf).tolist()

    # ── Feature Importance ──
    importances = rf_model.feature_importances_.tolist()
    feature_importance = dict(zip(FEATURE_NAMES, importances))
    feature_importance_sorted = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Save Models ──
    print("[*] Saving models and metrics...")
    joblib.dump(rf_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(lr_model, os.path.join(model_dir, 'lr_model.pkl'))
    joblib.dump(iso_model, os.path.join(model_dir, 'iso_model.pkl'))

    # Save metrics
    metrics = {
        'random_forest': {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1_score': rf_f1,
            'auc_roc': rf_auc,
            'confusion_matrix': cm,
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'logistic_regression': {
            'accuracy': lr_accuracy
        },
        'feature_importance': feature_importance_sorted,
        'feature_names': FEATURE_NAMES
    }

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Model training complete!")
    print(f"    Models saved to: {model_dir}")
    return metrics


if __name__ == '__main__':
    train()
