import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/logs.csv"

def save_log(total, attacks, normal):
    log = {
        "Time": datetime.now(),
        "Total Traffic": total,
        "Attacks": attacks,
        "Normal": normal
    }

    df = pd.DataFrame([log])

    # Always write header if file doesn't exist
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)