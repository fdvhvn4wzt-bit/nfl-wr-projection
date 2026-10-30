# NFL WR Receiving Yard Projection

A small FastAPI web app with an iPad-friendly interface. It retrieves historical player statistics through the nflreadpy/nflverse package. It is designed for postgame/statistical use, not live game updates.

## Deploy on Render
1. Upload the contents of this folder to your GitHub repository.
2. In Render, create a **New Web Service** and connect that GitHub repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Choose a free instance if available, then deploy.
6. Open the HTTPS URL on iPad Safari. Use Share → Add to Home Screen.

## Notes
This is an initial, unverified prototype. nflverse data availability and column names can change; the app may need adjustment after its first deployment. Projection calculations are estimates, not betting advice. The defensive matchup calculation and sample weighting should be validated before relying on outputs. Early-season samples are especially volatile.
