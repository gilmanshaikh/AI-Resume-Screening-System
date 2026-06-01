@echo off
echo Starting Resume Screening App...
start "" http://localhost:8501
python -m streamlit run webapp.py --server.port 8501
pause
