@echo off
echo Starting AQI Prediction App...
streamlit run app.py --server.port 8501 --server.headless true
pause
