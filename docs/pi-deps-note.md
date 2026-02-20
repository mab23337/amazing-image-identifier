# Raspberry Pi Dependency Note (Python 3.13)

On Raspberry Pi (Python 3.13), pip fails installing torchvision==0.16.0 because no compatible cp313/aarch64 distribution exists.

Workaround for Epic 1 testing:
- Install backend-only deps: Flask, flask-cors, gunicorn
- Run backend scaffold using production_app.py (if AI deps optional) or app_lite.py (no AI imports)

To run full AI inference on Pi:
- Use Python 3.10/3.11 (more compatible torch/torchvision wheels), OR
- Update pinned torchvision version to one available for this Pi/Python environment.

## Logging (TR-32)
A reusable logging helper was added in logging_setup.py. It can be imported by the Flask app to provide console + rotating file logging.
