"""Isolated local backend for browser QA; never loads the user's ADAM data."""
from pathlib import Path
import os
import sys
root = Path(__file__).resolve().parents[2]
os.environ['ADAM_DATA_DIR'] = str(root / 'adam-desktop/artifacts/qa/shared-browser-data')
os.environ['ADAM_DISABLE_HARDWARE'] = '1'
sys.path.insert(0, str(root / 'adam-desktop/src'))
import backend
print(f'QA backend PID={os.getpid()}, URL=http://127.0.0.1:8669', flush=True)
backend.app.run(host='127.0.0.1', port=8669, threaded=True, use_reloader=False)
