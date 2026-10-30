from pyngrok import ngrok

try:
    public_url = ngrok.connect(5173).public_url
    print(f"==========================================")
    print(f"PUBLIC_NGROK_URL: {public_url}")
    print(f"==========================================")
except Exception as e:
    print(f"Error opening ngrok tunnel: {e}")
