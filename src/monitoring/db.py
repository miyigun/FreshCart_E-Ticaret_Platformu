# src/monitoring/db.py
import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

# Import hatasını çözmek için 'src' modül yolunu düzeltiyoruz
try:
    from src.config import PROCESSED_DATA_DIR
except ImportError:
    # Eğer doğrudan bu dosya çalıştırılırsa (python src/monitoring/db.py gibi)
    # üst dizini yola ekleyerek import etmeye çalış
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.config import PROCESSED_DATA_DIR

DB_PATH = PROCESSED_DATA_DIR / 'monitoring.db'

def get_connection():
    """Creates a SQLite database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    """Initializes the log table (if it does not exist)."""
    # Klasörün var olduğundan emin ol
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Prediction Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER,
            -- --- Log Critical Features (For Drift Detection) ---
            purchase_velocity REAL,
            days_since_last_order REAL,
            -- ---------------------------------------------------
            predicted_prob REAL,
            predicted_label INTEGER,
            actual_label INTEGER DEFAULT NULL, -- Ground truth to be updated later
            model_version TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Monitoring DB initialized at {DB_PATH}")

def log_prediction(user_id, features, prob, label, model_version='v1'):
    """Logs a prediction to the database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = '''
            INSERT INTO predictions 
            (user_id, purchase_velocity, days_since_last_order, predicted_prob, predicted_label, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        # features can be a dictionary or a dataframe row
        cursor.execute(query, (
            user_id,
            features.get('purchase_velocity', 0),
            features.get('days_since_last_order', 0),
            prob,
            int(label),
            model_version
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging prediction: {e}")

# Initialize DB (when running this file directly)
if __name__ == "__main__":
    init_db()