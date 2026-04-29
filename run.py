import webview
import threading
from app import app # Import your Flask 'app' object

def start_server():
    # Run the Flask app without the debugger
    app.run(port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # 1. Start Flask in a background thread
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    # 2. Create a native window pointing to your localhost
    webview.create_window('GNN DDoS Detector', 'http://127.0.0.1:5000')
    webview.start()