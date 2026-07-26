AQI PREDICTION SYSTEM — DRDO INTERNSHIP
========================================
Intern   : Kushagra Saxena, MAIT Delhi
Project  : Air Quality Index Prediction using ML
Duration : June–July 2026

SETUP (one-time, needs internet):
1. Install Anaconda from https://www.anaconda.com
2. Open Anaconda Prompt and run:
      pip install -r requirements.txt

HOW TO RUN (fully offline):
1. Copy this entire folder to DRDO system
2. Place city_day.csv in the same folder
3. Double-click RUN_APP.bat  (Windows)
   OR run:  streamlit run app.py
4. Open browser at:  http://localhost:8501

FOLDER STRUCTURE:
  app.py              ← Main Streamlit app
  requirements.txt    ← Python dependencies
  city_day.csv        ← Dataset (you must add this)
  RUN_APP.bat         ← Windows launcher
  RUN_APP.sh          ← Linux/Mac launcher
  README.txt          ← This file

PAGES IN APP:
  Dashboard         → Overview, city AQI map
  Predict AQI       → Enter pollutant values → get AQI
  Model Comparison  → R2, MAE, RMSE charts
  EDA & Plots       → Heatmaps, distributions
  Report            → Full project report
