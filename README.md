# Extracting-Insights-from-Customer-Feedback

Small Flask app for uploading and reviewing customer feedback.

## Quick start (Windows / PowerShell)

1. Create and activate a virtual environment

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r interninfos\requirements.txt
   ```

2. Copy example env and set secrets

   ```powershell
   copy interninfos\.env.example interninfos\.env
   # Edit interninfos\.env and fill your DB credentials and JWT_SECRET_KEY
   ```

3. Create the database (one-time)

   Use your MySQL client to import `schema.sql` or run the commands inside it..

4. Run the app

   ```powershell
   python run.py
   ```

   Open http://127.0.0.1:5000

## Notes

- Do not commit `.env` with real secrets — use `.env.example` instead.
- Prefer creating a non-root DB user for production.
