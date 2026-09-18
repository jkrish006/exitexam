# Shopping Mode Predictor (Flask)

Simple Flask app that predicts a user's preferred shopping mode (Online, In-Store, Hybrid) from shopping behavior inputs.

Files:
- `app.py` — Flask app and model load/train logic
- `templates/index.html` — input form
- `templates/result.html` — prediction result view
- `requirements.txt` — Python dependencies

Run (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.
