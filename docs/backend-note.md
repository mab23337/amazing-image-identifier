# Backend API (TR-26)

Backend is implemented using Flask.

- production_app.py is the main application.
- app_lite.py is a lightweight Flask API scaffold used to validate backend/CORS/logging on Raspberry Pi when AI dependencies are not available.

Run lite backend:
python app_lite.py

Then open:
http://<pi-ip>:5000
