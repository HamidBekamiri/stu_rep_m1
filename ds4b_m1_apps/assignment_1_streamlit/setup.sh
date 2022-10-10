mkdir -p ~/.streamlit/

echo "[theme]
primaryColor='#A68101'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#E8E4D7'
textColor='#262730'
font='sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml