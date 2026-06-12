@echo off
cd /d "%~dp0"
echo Starting TSLA Stock Price Prediction Dashboard...
echo.
echo Open this URL in your browser:
echo http://127.0.0.1:8501
echo.
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
echo.
echo Streamlit stopped. Press any key to close this window.
pause >nul
