import os
import json
from datetime import datetime
from file_utils import get_logs_dir

def write_log(event_name: str, payload: dict):
    """
    Writes a single log entry into Backend/logs.
    File format: YYYY-MM-DD.log
    Adds timestamp + event name + JSON payload.
    """

    os.makedirs(get_logs_dir(), exist_ok=True)

    log_path = os.path.join(
        get_logs_dir(),
        f"{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "event": event_name,
        "data": payload
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
